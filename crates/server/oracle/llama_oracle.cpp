// Differential-test oracle: answers tokenizer / chat-template / chat-parse questions with
// llama.cpp's own code (libllama, libllama-common, libllama-server-impl), loading the model
// GGUF with vocab_only (no weights, no inference). One JSON request per stdin line, one JSON
// response per stdout line.
//
//   {"op":"tokenize","text":"..","add_special":false,"parse_special":true}  -> {"ids":[..]}
//   {"op":"detokenize","ids":[..],"special":true}                           -> {"text":".."}
//   {"op":"apply","body":{<chat body>},"kwargs":{..}}                        -> {"prompt":"..","format":"..","generation_prompt":"..","stops":[..]}
//   {"op":"parse","body":{<chat body>},"kwargs":{..},"output":".."}          -> {"diffs":[[..per token..]],"final":{msg},"final_diffs":[..]}
//
// "kwargs" are server-level --chat-template-kwargs (e.g. {"enable_thinking":false}).
#include "llama.h"
#include "common.h"
#include "chat.h"
#include "server-common.h"
#include <cstdio>
#include <iostream>
#include <string>

using json = common_json;

static json diff_json(const common_chat_msg_diff & d) {
    json j = json::object();
    if (!d.reasoning_content_delta.empty()) j["reasoning_content"] = d.reasoning_content_delta;
    if (!d.content_delta.empty()) j["content"] = d.content_delta;
    if (d.tool_call_index != std::string::npos) {
        json tc;
        tc["index"] = d.tool_call_index;
        if (!d.tool_call_delta.id.empty()) { tc["id"] = d.tool_call_delta.id; tc["type"] = "function"; }
        json f = json::object();
        if (!d.tool_call_delta.name.empty()) f["name"] = d.tool_call_delta.name;
        if (!d.tool_call_delta.arguments.empty()) f["arguments"] = d.tool_call_delta.arguments;
        if (!f.empty()) tc["function"] = f;
        j["tool_calls"] = json::array({tc});
    }
    return j;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf < requests.jsonl\n", argv[0]); return 2; }
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tmpls = common_chat_templates_init(model, "");

    std::string line;
    while (std::getline(std::cin, line)) {
        json out;
        try {
            json req = json::parse(line);
            std::string op = req.at("op");
            if (op == "tokenize") {
                auto ids = common_tokenize(vocab, req.at("text").get<std::string>(),
                    req.value("add_special", false), req.value("parse_special", true));
                out["ids"] = ids;
            } else if (op == "detokenize") {
                std::vector<llama_token> ids;
                for (size_t i = 0; i < req.at("ids").size(); i++) ids.push_back(req.at("ids").at(i).get<int>());
                out["text"] = common_detokenize(vocab, ids, req.value("special", true));
                std::string pieces;
                json pj = json::array();
                for (auto t : ids) pj.push_back(common_token_to_piece(vocab, t, req.value("special", true)));
                out["pieces"] = pj;
            } else if (op == "apply" || op == "parse") {
                server_chat_params opt;
                opt.use_jinja = true;
                opt.prefill_assistant = true;
                opt.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
                opt.allow_image = opt.allow_audio = opt.allow_video = false;
                opt.enable_thinking = true;
                if (req.contains("kwargs")) {
                    for (const auto & it : req["kwargs"].items()) opt.chat_template_kwargs[it.key()] = it.value().dump();
                    // server main does the same for the CLI default
                    auto et = opt.chat_template_kwargs.find("enable_thinking");
                    if (et != opt.chat_template_kwargs.end()) opt.enable_thinking = et->second == "true";
                }
                opt.tmpls = common_chat_templates_init(model, "");
                std::vector<raw_buffer> files;
                common_chat_session session;
                json body = req.at("body");
                json params = oaicompat_chat_params_parse(vocab, body, opt, files, session);
                out["prompt"] = params.at("prompt");
                out["format"] = common_chat_format_name(session.format());
                out["generation_prompt"] = session.generation_prompt();
                out["stops"] = session.additional_stops();
                out["thinking_start"] = session.thinking_start_tag();
                out["thinking_end"] = session.thinking_end_tags();
                if (op == "parse") {
                    std::string text = req.at("output");
                    auto toks = common_tokenize(vocab, text, false, true);
                    common_chat_msg prev = session.msg();
                    std::vector<std::string> ids_cache;
                    int idn = 0;
                    auto gen = [&]() { return std::string("call") + std::to_string(idn++); };
                    json steps = json::array();
                    for (auto t : toks) {
                        common_chat_input chunk;
                        chunk.append(common_token_to_piece(vocab, t, true), t);
                        common_chat_msg m = session.feed(chunk);
                        if (m.empty()) { steps.push_back(json::array()); continue; }
                        m.set_tool_call_ids(ids_cache, gen);
                        json dj = json::array();
                        for (auto & d : common_chat_msg_diff::compute_diffs(prev, m)) dj.push_back(diff_json(d));
                        steps.push_back(dj);
                        prev = m;
                    }
                    common_chat_msg m = session.finish();
                    json fd = json::array();
                    if (!m.empty()) {
                        m.set_tool_call_ids(ids_cache, gen);
                        for (auto & d : common_chat_msg_diff::compute_diffs(prev, m)) fd.push_back(diff_json(d));
                    } else {
                        m = prev;
                    }
                    out["steps"] = steps;
                    out["final_diffs"] = fd;
                    out["final"] = m.to_json_oaicompat();
                }
            } else {
                out["error"] = "unknown op";
            }
        } catch (const std::exception & e) {
            out["error"] = e.what();
        }
        std::cout << out.dump() << std::endl;
    }
    llama_model_free(model);
    return 0;
}
