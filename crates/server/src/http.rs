//! hyper (HTTP/1.1) front end: body collection, routing into [`crate::service`], SSE streaming.

use crate::engine::Engine;
use crate::service::{handle, Reply, State};
use bytes::Bytes;
use hyper::body::{Body, Frame, Incoming};
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::net::TcpListener;
use tokio::sync::mpsc;

/// Response body: either one buffer or a channel of SSE frames.
pub enum RBody {
    Full(Option<Bytes>),
    Chan(mpsc::Receiver<String>),
}

impl Body for RBody {
    type Data = Bytes;
    type Error = Infallible;
    fn poll_frame(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        match self.get_mut() {
            RBody::Full(b) => Poll::Ready(b.take().map(|b| Ok(Frame::data(b)))),
            RBody::Chan(rx) => rx.poll_recv(cx).map(|o| o.map(|s| Ok(Frame::data(Bytes::from(s))))),
        }
    }
}

const MAX_BODY: usize = 64 << 20;

async fn collect(mut b: Incoming) -> Result<Vec<u8>, String> {
    use std::future::poll_fn;
    let mut out = Vec::new();
    loop {
        let f = poll_fn(|cx| Pin::new(&mut b).poll_frame(cx)).await;
        match f {
            None => return Ok(out),
            Some(Err(e)) => return Err(e.to_string()),
            Some(Ok(fr)) => {
                if let Ok(d) = fr.into_data() {
                    out.extend_from_slice(&d);
                    if out.len() > MAX_BODY {
                        return Err("request body too large".into());
                    }
                }
            }
        }
    }
}

fn json_resp(status: u16, v: &serde_json::Value) -> Response<RBody> {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json; charset=utf-8")
        .header("Access-Control-Allow-Origin", "*")
        .body(RBody::Full(Some(Bytes::from(serde_json::to_vec(v).unwrap()))))
        .unwrap()
}

pub async fn serve<E: Engine>(st: Arc<State<E>>, listener: TcpListener) -> std::io::Result<()> {
    loop {
        let (sock, _) = listener.accept().await?;
        let _ = sock.set_nodelay(true);
        let st = st.clone();
        tokio::spawn(async move {
            let svc = service_fn(move |req: Request<Incoming>| {
                let st = st.clone();
                async move {
                    let method = req.method().as_str().to_string();
                    let path = req.uri().path().to_string();
                    let query = req.uri().query().unwrap_or("").to_string();
                    if method == "OPTIONS" {
                        let r = Response::builder()
                            .status(204)
                            .header("Access-Control-Allow-Origin", "*")
                            .header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                            .header("Access-Control-Allow-Headers", "*")
                            .body(RBody::Full(None))
                            .unwrap();
                        return Ok::<_, Infallible>(r);
                    }
                    let body = match collect(req.into_body()).await {
                        Ok(b) => b,
                        Err(e) => return Ok(json_resp(400, &crate::api::error_json(400, &e))),
                    };
                    let reply = handle(st, &method, &path, &query, &body).await;
                    Ok(match reply {
                        Reply::Json(code, v) => json_resp(code, &v),
                        Reply::Stream(rx) => Response::builder()
                            .status(200)
                            .header("Content-Type", "text/event-stream")
                            .header("Cache-Control", "no-cache")
                            .header("Access-Control-Allow-Origin", "*")
                            .body(RBody::Chan(rx))
                            .unwrap(),
                    })
                }
            });
            let _ = hyper::server::conn::http1::Builder::new().keep_alive(true).serve_connection(TokioIo::new(sock), svc).await;
        });
    }
}
