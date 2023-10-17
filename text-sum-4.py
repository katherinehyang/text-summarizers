import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device('cpu')

text ="""
Former President Donald Trump said in a social media post he’s been informed by special counsel Jack Smith that he is a target of the criminal investigation into efforts to overturn the 2020 election, a sign he may soon be charged by the special counsel. “Deranged Jack Smith, the prosecutor with Joe Biden’s DOJ, sent a letter (again, it was Sunday night!) stating that I am a TARGET of the January 6th Grand Jury investigation, and giving me a very short 4 days to report to the Grand Jury, which almost always means an Arrest and Indictment,” Trump posted on Truth Social. Trump’s attorneys, including Todd Blanche, received the target letter from Smith’s team Sunday informing them that their client could face charges in the investigation into efforts to overturn the 2020 election, two sources familiar with what happened tell CNN. A target letter from federal prosecutors to Trump makes clear that prosecutors are focused on the former president’s actions in the investigation into overturning the 2020 election – and not just of those around him who tried to stop his election loss. Trump’s legal team has not formally responded to the invitation to testify before the grand jury, which the letter provides, but it is largely expected that Trump will decline to do so. The letter caught Trump’s team off guard, who had not been anticipating Smith to potentially bring charges this month, or against Trump. Trump’s advisers spent Tuesday morning calling lawyers and allies, trying to figure out who else – if anyone – received a target letter related to the special counsel’s investigation into the aftermath of the 2020 election, multiple sources familiar with the outreach told CNN. Trump’s advisers are hoping to glean better insight into what a potential criminal case against the former president might entail, according to sources. So far, Trump’s team has not identified anyone else who got a target letter, the sources said. Trump also received a target letter from Smith in May roughly three weeks before he was indicted in the investigation into the mishandling of classified documents. Smith declined to comment Tuesday when asked by CNN about the target letter and whether his office is preparing to indict the former president. CNN spotted Smith leaving a Subway sandwich shop in Washington, DC. The White House declined to comment. CNN has also reached out to Biden’s reelection campaign for comment. Trump has already been indicted twice this year. Manhattan District Attorney Alvin Bragg charged the former president on 34 counts of falsifying business records in March, and Smith charged Trump on 37 counts in the classified documents investigation last month. Trump pleaded not guilty in both cases. Justice Department regulations allow for prosecutors to notify subjects of an investigation that they have become a target. Often a notification that a person is a target is a strong sign an indictment could follow, but it is possible the recipient is not ultimately charged. Those notifications aren’t required, but prosecutors have the discretion to notify subjects that they have become a target. While Trump didn’t say specifically why he was told to report to the grand jury, individuals who receive a target letter typically are given the chance to appear before a grand jury to present evidence or testify if they choose. In the classified documents case, Trump received a target letter from the special counsel’s office on May 19."""


preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


# summmarize 
summary_ids = model.generate(tokenized_text, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=100, early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)