import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import rouge
import numpy as np

sample_text = "(CNN) -- The owner of a taxidermy school in Blanco, Texas, says he's been receiving more phone calls and attention than he'd like over the identity of a dead roughly 30-pound, mostly hairless coyote-like creature. This strange creature has sparked a media frenzy in Blanco, Texas. \"I don't know what it is. ... I do know that I have an odd animal,\" Jerry Ayer said Thursday. He said word spread quickly that he was in possession of an unknown animal -- often speculated in his region to be chupacabras, or mythical creatures, he said. Soon, the local, national and even international media picked up on what he said could be a \"genetically defective coyote.\" \"Chupacabra\" roughly translates from Spanish to \"goat sucker.\" Reported victims are said to have puncture wounds to their necks, supposedly where the chupacabra drained their blood.  Watch people examine mystery creature » . \"I don't believe in the chupacabra,\" Ayer said, adding that he's been in the midst of a \"media blitz\" -- receiving 50 phone calls from media outlets and citizens, as well as death threats late at night -- since word got out about the strange beast. \"It's been rough on me,\" he said. \"I'm almost at the point where I'm going to take my sign down and hide under a rock somewhere.\" He said he received the body from a former student whose cousin owns the barn where the creature succumbed to poison left for rodents. Before he knew it, he said, word spread that he had a chupacabra. Ayers, who doesn't hunt and regards himself as a wildlife artist, said he intends to stuff and mount the animal. He said Texas A&M University has taken tissue samples to determine exactly what it is, and other universities have also reached out to him. \"It\'ll probably end up in a museum,\" he said. He\'s hopeful the attention will soon die down. \"There\'s no way I could teach with this,\" he said, explaining that his school, Blanco Taxidermy School, generally gives one-on-one instruction in the town of about 1,500 people about 50 miles west of Austin. \"I\'m unable to do any of that right now just because of the media frenzy,\" he said."
summary = "Word quickly spread about coyote-like body at Texas school. Owner says he doubts it's a legendary chupacabra. He has gotten dozens of calls from media, residents. Body came from former student; it was killed in barn by poison."

print("---Text---")
print(sample_text)
print("---Original Summary---")
print(summary)

print ("---TextRank Model Predicted Summary---")



print("---Seq2seq Model Predicted Summary---")


model_path = "/models/"
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
output_dir = "/Users/Maksim.Nevar/PycharmProjects/model_comparison/models/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_ids = tokenizer(sample_text, max_length=512, truncation=True,
                   padding='max_length', return_tensors='pt').to(device)
summaries = model.generate(input_ids=input_ids['input_ids'],
                           attention_mask=input_ids['attention_mask'],
                           max_length=124)
decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True)
                    for s in summaries]

print("---Fine-Tuned Bart Model Predicted Summary---")
print(decoded_summaries[0])


def evaluate_summary(summary_test, predicted):
    global score1Total, score2Total, scoreLTotal
    rouge_score = rouge.Rouge()
    scores = rouge_score.get_scores(summary_test, predicted, avg=True)
    score_1 = scores['rouge-1']['f']
    score_2 = scores['rouge-2']['f']
    score_L = scores['rouge-l']['f']

    print("---Original Summary---")
    print(summary_test)
    print("---Predicted Summary---")
    print(predicted)
    print("---Rouge evaluation for predicted summary---")
    print("Rouge1: ", round(score_1,2), "| Rouge2: ", round(score_2,2), "| RougeL: ",
    round(score_L,2), "--> Avg Rouge:", round(np.mean([score_1,score_2,score_L]), 2))

evaluate_summary(summary, decoded_summaries[0])