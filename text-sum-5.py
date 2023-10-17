from transformers import PegasusTokenizer, PegasusForConditionalGeneration

def generate_summary(text, model, tokenizer):
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    
    # Generate summary
    summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True)
    
    # Decode the summary tokens back to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load the Pegasus model and tokenizer
model_name = "google/pegasus-large"  # You can try using other Pegasus model sizes, such as "google/pegasus-xsum"
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)

# Sample input text
text = """
    John G. Roberts, Jr. We'll hear argument first this morning in Case 21-476, 303 Creative LLC versus Elenis. Ms. Waggoner. Kristen Kellie Waggoner Mr. Chief Justice, and may it please the Court: Lorie Smith blends art with technology to create custom messages using words and graphics. She serves all people, deciding what to create based on the message, not who requests it. But Colorado declares her speech a public accommodation and insists that she create and speak messages that violate her conscience. This Court rejects such government-compelled speech. In Hurley, the Court considered a very similar issue, asking two questions: Is there speech, and is the message affected? That test is easily met here. Colorado agrees Ms. Smith creates speech, and the law undeniably affects her message. She's not asking this Court to create new law but to apply its precedent. Colorado first says this case is about a sale. It's not just about a sale. The state forces Ms. Smith to create speech, not simply sell it. Next, Colorado says it can compel speech on the same topic. But Ms. Smith believes opposite-sex marriage honors scripture and same-sex marriage contradicts it. If the government can label this speech equivalent, it can do so for any speech, whether religious or political. Under Colorado's theory, jurisdictions could force a Democrat publicist to write a Republican's press release. Colorado's last resort is to argue that it can at least compel the same expression. But even the same expression can mean different things, like a black sculptor who carves a custom cross to celebrate a Catholic baptism but not an Aryan church rally. If the government may not force motorists to display a motto, school children to say a pledge, or parades to include banners, Colorado may not force Ms. Smith to create and speak messages on pain of investigation, fine, and re-education. I welcome this Court's questions. Clarence Thomas Counsel, would you spend just a few minutes on whether or not this -- your case is ripe? Kristen Kellie Waggoner Sure. This Court has considered pre-enforcement challenges before, and, in those contexts, it has looked at the facts. This is one of the strongest pre-enforcement cases, I think, that this Court has considered in that the parties have stipulated every message that Ms. Smith would create has a unique, customized message and that it celebrates a wedding and celebrates a marriage. It's also difficult to imagine a scenario where there is a more aggressive enforcement history by Colorado. Ms. Smith's speech has been chilled. For six years, she has been unable to speak in the marketplace. She's ready to do so today, and she's ready to post her website statement today, which makes this case ripe.
"""

# Generate summary for the input text
summary = generate_summary(text, model, tokenizer)

# Print the generated summary
print("Generated Summary:", summary)
