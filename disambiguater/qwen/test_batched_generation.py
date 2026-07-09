import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
print("Model weight loaded in ", str(model.dtype))
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.config.use_cache = True
# model.set_attn_implementation("flash_attention_2")
# print(model.config._attn_implementation)

short_text = ('You are an expert computational linguist specializing in Uyghur morphosyntax and contextual '
              'disambiguation. Disambiguate the marked Uyghur word by choosing the analysis whose features are ALL '
              'correct in context.\nSentence: <t> ۋوگزال </t> سۇپىسى ئۇزاتقۇچىلار بىلەن تولۇپ كەتكەنىدى.\nWord: '
              'ۋوگزال\nCandidates:\n1. Lemma: ۋوگزال | Features: Noun, Nominative\n2. Lemma: ۋوگزال | Features: Noun, '
              'Nominative + Combined with: Lemma: ئى | Features: Copula, Aorist, Third person, Singular')

long_text = ('You are an expert computational linguist specializing in Uyghur morphosyntax and contextual '
             'disambiguation. Disambiguate the marked Uyghur word by choosing the analysis whose features are ALL '
             'correct in context.\nSentence: ھەتتا ياۋا <t> توشقانمۇ </t> ئۇنى كۆكرىكى بىلەن تۈرتۈپ، ئۇنىڭ ئاستىدا '
             'بىمالال ھەرىكەتلىنەلەيتتى.\nWord: توشقانمۇ\nCandidates:\n1. Lemma: توش | Features: Standard verb, '
             'Intransitive, Past absolute gerund, Nominative + Combined with: Lemma: ئى | Features: Copula, Aorist, '
             'Third person, Singular + Combined with: Lemma: مۇ | Features: Post-adverb\n2. Lemma: توشقان | Features: '
             'Noun, Nominative + Combined with: Lemma: ئى | Features: Copula, Aorist, Third person, Singular + '
             'Combined with: Lemma: مۇ | Features: Post-adverb\n3. Lemma: توشقان | Features: Adjective, Substantive, '
             'Nominative + Combined with: Lemma: ئى | Features: Copula, Aorist, Third person, Singular + Combined '
             'with: Lemma: مۇ | Features: Post-adverb\n4. Lemma: توش | Features: Standard verb, Intransitive, '
             'Relative substantival verbal adjective, Substantive, Nominative + Combined with: Lemma: ئى | Features: '
             'Copula, Aorist, Third person, Singular + Combined with: Lemma: مۇ | Features: Post-adverb\n5. Lemma: '
             'توشقان | Features: Adjective + Combined with: Lemma: مۇ | Features: Post-adverb\n6. Lemma: توش | '
             'Features: Standard verb, Intransitive, Relative substantival verbal adjective, Nominative + Combined '
             'with: Lemma: مۇ | Features: Post-adverb\n7. Lemma: توش | Features: Standard verb, Intransitive, '
             'Relative substantival verbal adjective, Substantive, Nominative + Combined with: Lemma: مۇ | Features: '
             'Post-adverb\n8. Lemma: توش | Features: Standard verb, Intransitive, Past absolute gerund, Nominative + '
             'Combined with: Lemma: مۇ | Features: Post-adverb\n9. Lemma: توشقان | Features: Noun, Nominative + '
             'Combined with: Lemma: مۇ | Features: Post-adverb\n10. Lemma: توشقان | Features: Adjective, Adverbial + '
             'Combined with: Lemma: مۇ | Features: Post-adverb\n11. Lemma: توش | Features: Standard verb, '
             'Intransitive, Past, Third person, Singular + Combined with: Lemma: مۇ | Features: Post-adverb\n12. '
             'Lemma: توشقان | Features: Adjective, Substantive, Nominative + Combined with: Lemma: مۇ | Features: '
             'Post-adverb\n13. Lemma: توش | Features: Standard verb, Intransitive, Relative substantival verbal '
             'adjective, Nominative + Combined with: Lemma: ئى | Features: Copula, Aorist, Third person, Singular + '
             'Combined with: Lemma: مۇ | Features: Post-adverb\n')


def test(short, long, min, max):
    short_messages = [{"role": "user", "content": short}]
    long_messages = [{"role": "user", "content": long}]

    short = tok.apply_chat_template(short_messages, tokenize=False, add_generation_prompt=True)
    long = tok.apply_chat_template(long_messages, tokenize=False, add_generation_prompt=True)

    # (A) unpadded, batch size 1
    enc1 = tok(short, return_tensors="pt").to("cuda")

    # (B) same sentence, but LEFT-padded inside a batch sized 2 (padding forced by the long seq)
    enc2 = tok([short, long], return_tensors="pt", padding=True).to("cuda")
    print(f"n_pad_tokens_added = {(enc2['attention_mask'][0] == 0).sum().item()}")

    with torch.inference_mode():
        for mnt in range(min, max):
            g1 = model.generate(**enc1, max_new_tokens=mnt, do_sample=False)
            g2 = model.generate(**enc2, max_new_tokens=mnt, do_sample=False)
            unpadded = tok.decode(g1[0, enc1['input_ids'].shape[1]:], skip_special_tokens=True)
            padded = tok.decode(g2[0, enc2['input_ids'].shape[1]:], skip_special_tokens=True)
            if unpadded != padded:
                print("Different Generation when max_new_tokens=", mnt)
                print()
                return


for i in range(1950, 1, -3):  # range adjusted to the two texts
    # len(tok("ANSWER: 3<|eos|>")) === 6, adjust to higher value to mock CoT mode
    test(short_text, long_text[:-i], 6, 7)