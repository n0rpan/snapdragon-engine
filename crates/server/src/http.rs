//! hyper (HTTP/1.1) front end: access checks, body collection, routing into [`crate::service`],
//! SSE streaming.
//!
//! Access (REVIEW-v0.1.1 B1): the backend port is for local clients only. No CORS headers are
//! sent; a request whose `Host` is not a loopback name with the bound port, or whose `Origin` is
//! not loopback (`Sec-Fetch-Site: cross-site` too), gets 403 before its body is read. With an API
//! key (`--api-key-file` / `OXE_API_KEY_FILE`) every route, `/health` included (the controller
//! sends the key on its health polls too), needs `Authorization: Bearer <key>` (or `X-Api-Key`),
//! compared in constant time; else 401.

use crate::engine::Engine;
use crate::service::{handle_session, Reply, State};
use bytes::Bytes;
use hyper::body::{Body, Frame, Incoming};
use hyper::service::service_fn;
use hyper::header::HeaderMap;
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
    let mut r = Response::builder().status(status).header("Content-Type", "application/json; charset=utf-8");
    if status == 401 {
        r = r.header("WWW-Authenticate", "Bearer");
    }
    r.body(RBody::Full(Some(Bytes::from(serde_json::to_vec(v).unwrap())))).unwrap()
}

/// Who may talk to this port (B1). Built once at startup; `port` is the bound port.
#[derive(Clone, Debug, Default)]
pub struct Access {
    pub port: u16,
    /// extra accepted Host names (lower case, without port): `--allow-host`, or the specific
    /// non-loopback address given with `--host`
    pub extra_hosts: Vec<String>,
    /// accepted API keys (empty = no key needed)
    pub keys: Vec<Vec<u8>>,
}

const LOOPBACK_NAMES: [&str; 3] = ["127.0.0.1", "localhost", "[::1]"];

/// Split `name[:port]` (IPv6 in brackets) into a lower-case name and the port text.
fn split_host_port(h: &str) -> (String, Option<&str>) {
    let h = h.trim();
    let (name, port) = if h.starts_with('[') {
        match h.find(']') {
            Some(e) => (&h[..=e], h[e + 1..].strip_prefix(':')),
            None => (h, None),
        }
    } else {
        match h.rsplit_once(':') {
            Some((n, p)) => (n, Some(p)),
            None => (h, None),
        }
    };
    (name.to_ascii_lowercase(), port)
}

/// Constant-time byte comparison (the length is not secret; the content is).
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut d = 0u8;
    for i in 0..a.len().max(b.len()) {
        d |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0xff);
    }
    std::hint::black_box(d) == 0 && a.len() == b.len()
}

impl Access {
    fn host_ok(&self, host: &str) -> bool {
        let (name, port) = split_host_port(host);
        let port_ok = match port {
            Some(p) => p.parse::<u16>().ok() == Some(self.port),
            None => self.port == 80,
        };
        port_ok && (LOOPBACK_NAMES.contains(&name.as_str()) || self.extra_hosts.iter().any(|h| *h == name))
    }

    /// `Origin` is same-site when its host is loopback (any port: the controller's own pages on
    /// 8092 are local) or an extra host; `null` and anything else is cross-site.
    fn origin_ok(&self, origin: &str) -> bool {
        let o = origin.trim();
        let Some(rest) = o.strip_prefix("http://").or_else(|| o.strip_prefix("https://")) else {
            return false;
        };
        if rest.contains('/') || rest.contains('@') {
            return false;
        }
        let (name, port) = split_host_port(rest);
        if let Some(p) = port {
            if p.parse::<u16>().is_err() {
                return false;
            }
        }
        LOOPBACK_NAMES.contains(&name.as_str()) || self.extra_hosts.iter().any(|h| *h == name)
    }

    fn key_ok(&self, headers: &HeaderMap) -> bool {
        let bearer = headers
            .get_all("authorization")
            .iter()
            .filter_map(|v| v.to_str().ok())
            .filter_map(|v| {
                let (scheme, tok) = v.trim().split_once(' ')?;
                scheme.eq_ignore_ascii_case("bearer").then(|| tok.trim().as_bytes().to_vec())
            });
        let xkey = headers.get_all("x-api-key").iter().map(|v| v.as_bytes().to_vec());
        let given: Vec<Vec<u8>> = bearer.chain(xkey).collect();
        // check every (given, key) pair so the time does not depend on which key matched
        let mut ok = false;
        for g in &given {
            for k in &self.keys {
                ok |= ct_eq(g, k);
            }
        }
        ok
    }

    /// `Err((status, message))` when the request must be refused before routing.
    pub fn check(&self, method: &str, path: &str, headers: &HeaderMap) -> Result<(), (u16, &'static str)> {
        let hosts: Vec<&[u8]> = headers.get_all("host").iter().map(|v| v.as_bytes()).collect();
        if hosts.len() > 1 {
            return Err((403, "more than one Host header"));
        }
        if let Some(h) = hosts.first() {
            match std::str::from_utf8(h) {
                Ok(h) if self.host_ok(h) => {}
                _ => return Err((403, "Host is not this loopback server (use 127.0.0.1, localhost or [::1] with the server port)")),
            }
        }
        for o in headers.get_all("origin") {
            if !o.to_str().map(|o| self.origin_ok(o)).unwrap_or(false) {
                return Err((403, "cross-site requests are refused"));
            }
        }
        for s in headers.get_all("sec-fetch-site") {
            if s.as_bytes().eq_ignore_ascii_case(b"cross-site") {
                return Err((403, "cross-site requests are refused"));
            }
        }
        let _ = (method, path);
        if !self.keys.is_empty() && !self.key_ok(headers) {
            return Err((401, "Invalid API Key"));
        }
        Ok(())
    }
}

/// Parse an API key file (llama-server's `--api-key-file` format): every non-empty line,
/// whitespace trimmed, is a key. An empty result is an error (fail closed).
pub fn parse_key_file(text: &str) -> Result<Vec<Vec<u8>>, String> {
    let keys: Vec<Vec<u8>> = text.lines().map(str::trim).filter(|l| !l.is_empty()).map(|l| l.as_bytes().to_vec()).collect();
    if keys.is_empty() {
        return Err("API key file holds no key".into());
    }
    Ok(keys)
}

/// Serve with the default access rules for the listener's port (no key).
pub async fn serve<E: Engine>(st: Arc<State<E>>, listener: TcpListener) -> std::io::Result<()> {
    let port = listener.local_addr()?.port();
    serve_with(st, listener, Access { port, ..Default::default() }).await
}

pub async fn serve_with<E: Engine>(st: Arc<State<E>>, listener: TcpListener, access: Access) -> std::io::Result<()> {
    let access = Arc::new(access);
    loop {
        let (sock, _) = listener.accept().await?;
        let _ = sock.set_nodelay(true);
        let st = st.clone();
        let access = access.clone();
        tokio::spawn(async move {
            let svc = service_fn(move |req: Request<Incoming>| {
                let st = st.clone();
                let access = access.clone();
                async move {
                    let method = req.method().as_str().to_string();
                    let path = req.uri().path().to_string();
                    let query = req.uri().query().unwrap_or("").to_string();
                    if let Err((code, msg)) = access.check(&method, &path, req.headers()) {
                        return Ok::<_, Infallible>(json_resp(code, &crate::api::error_json(code, msg)));
                    }
                    if method == "OPTIONS" {
                        // no CORS: a browser preflight from another site fails here
                        let r = Response::builder().status(204).header("Allow", "GET, POST, OPTIONS").body(RBody::Full(None)).unwrap();
                        return Ok::<_, Infallible>(r);
                    }
                    // CACHE-PERFECT: the controller's session hint (a hint, never a cache key)
                    let session = req.headers().get("x-oxe-session").and_then(|v| v.to_str().ok()).map(|s| s.chars().take(128).collect::<String>()).filter(|s| !s.is_empty());
                    let body = match collect(req.into_body()).await {
                        Ok(b) => b,
                        Err(e) => return Ok(json_resp(400, &crate::api::error_json(400, &e))),
                    };
                    let reply = handle_session(st, &method, &path, &query, &body, session).await;
                    Ok(match reply {
                        Reply::Json(code, v) => json_resp(code, &v),
                        Reply::Stream(rx) => Response::builder()
                            .status(200)
                            .header("Content-Type", "text/event-stream")
                            .header("Cache-Control", "no-cache")
                            .body(RBody::Chan(rx))
                            .unwrap(),
                    })
                }
            });
            let _ = hyper::server::conn::http1::Builder::new().keep_alive(true).serve_connection(TokioIo::new(sock), svc).await;
        });
    }
}
