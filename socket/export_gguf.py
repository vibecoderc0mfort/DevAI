"""Сохраняет веса Socket 1.0 в формат GGUF, который понимает llama.cpp (архитектура llama).

Запуск: python export_gguf.py weights.npz socket-1.0.gguf
"""
import json, sys
import numpy as np
import gguf
from model import Config, V, UNK, BOS, EOS, SPACE_ID

src, dst = sys.argv[1], sys.argv[2]
cfg = Config(**json.load(open(sys.argv[3]))) if len(sys.argv) > 3 else Config(d=128, layers=4, heads=4, ff=384, ctx=1024)
z = np.load(src)

w = gguf.GGUFWriter(dst, "llama")
w.add_name("Socket 1.0")
w.add_description("Socket 1.0: учебная нейросеть (около 0,9 млн параметров), написанная и обученная с нуля. "
                  "По короткому английскому описанию пишет небольшие функции на Python.")
w.add_author("DevAI")
w.add_context_length(cfg.ctx)
w.add_embedding_length(cfg.d)
w.add_block_count(cfg.layers)
w.add_feed_forward_length(cfg.ff)
w.add_head_count(cfg.heads)
w.add_head_count_kv(cfg.heads)
w.add_rope_dimension_count(cfg.dh)
w.add_rope_freq_base(cfg.base)
w.add_layer_norm_rms_eps(cfg.eps)
w.add_vocab_size(V)
w.add_file_type(gguf.LlamaFileType.ALL_F32)

# Словарь: <unk>, <s>, </s>, 256 байтов и «▁» (пробел). Всё остальное разбирается побайтово.
tokens, scores, types = [b"<unk>", b"<s>", b"</s>"], [0.0] * 3, [gguf.TokenType.UNKNOWN, gguf.TokenType.CONTROL, gguf.TokenType.CONTROL]
for b in range(256):
    tokens.append(f"<0x{b:02X}>".encode()); scores.append(0.0); types.append(gguf.TokenType.BYTE)
tokens.append("▁".encode()); scores.append(0.0); types.append(gguf.TokenType.NORMAL)
assert len(tokens) == V and tokens[SPACE_ID] == "▁".encode()
w.add_tokenizer_model("llama")
w.add_token_list(tokens)
w.add_token_scores(scores)
w.add_token_types(types)
w.add_unk_token_id(UNK)
w.add_bos_token_id(BOS)
w.add_eos_token_id(EOS)
w.add_add_bos_token(False)
w.add_add_eos_token(False)
w.add_add_space_prefix(False)
w.add_remove_extra_whitespaces(False)

f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
w.add_tensor("token_embd.weight", f32(z["E"]))            # выходной слой общий с входным, отдельного output.weight нет
w.add_tensor("output_norm.weight", f32(z["gf"]))
for i in range(cfg.layers):
    w.add_tensor(f"blk.{i}.attn_norm.weight", f32(z[f"g1.{i}"]))
    w.add_tensor(f"blk.{i}.ffn_norm.weight", f32(z[f"g2.{i}"]))
    # в llama.cpp матрицы хранятся как (выход, вход); у нас (вход, выход), поэтому транспонируем
    w.add_tensor(f"blk.{i}.attn_q.weight", f32(z[f"Wq.{i}"].T))
    w.add_tensor(f"blk.{i}.attn_k.weight", f32(z[f"Wk.{i}"].T))
    w.add_tensor(f"blk.{i}.attn_v.weight", f32(z[f"Wv.{i}"].T))
    w.add_tensor(f"blk.{i}.attn_output.weight", f32(z[f"Wo.{i}"].T))
    w.add_tensor(f"blk.{i}.ffn_gate.weight", f32(z[f"Wg.{i}"].T))
    w.add_tensor(f"blk.{i}.ffn_up.weight", f32(z[f"Wu.{i}"].T))
    w.add_tensor(f"blk.{i}.ffn_down.weight", f32(z[f"Wd.{i}"].T))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("записано:", dst)
