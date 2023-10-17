import os
import logging

from flask import Flask
from flask import request
from flask import jsonify
from transformers import pipeline

max_length = os.environ.get('MAX_LENGTH', 100)
min_length = os.environ.get('MIN_LENGTH', 5)

logging.basicConfig(level=logging.INFO) # INFO level for log

app = Flask(__name__)

# Bidirectional and Auto-Regressive (BART) model
summarizer = pipeline("summarization")

@app.route("/ping")
def ping():
    return {'success': True} # success message can be anything

@app.route('/summarize', methods=['POST'])
def summarize():
    app.logger.info("Loading summarizer...") 
    text = request.json['text'] # JSON input {'text': 'input text'}
    app.logger.info("Loaded summarizer.") 
    summary_text = summarizer(text, max_length, min_length, do_sample=False)[0]['summary_text']
    app.logger.info("Completed summarization.") 
    return jsonify({'summary': summary_text})

if __name__ == '__main__':
    app.run()





# text = """One month after the United States began what has become a troubled rollout of a national COVID vaccination campaign, the effort is finally gathering real steam. 
# Close to a million doses -- over 951,000, to be more exact -- made their way into the arms of Americans in the past 24 hours, the U.S. Centers for Disease Control and Prevention 
# reported Wednesday. That's the largest number of shots given in one day since the rollout began and a big jump from the previous day, when just under 340,000 doses were given, CBS News 
# reported. That number is likely to jump quickly after the federal government on Tuesday gave states the OK to vaccinate anyone over 65 and said it would release all the doses of 
# vaccine it has available for distribution. Meanwhile, a number of states have now opened mass vaccination sites in an effort to get larger numbers of people inoculated, CBS News 
# reported."""

# 'If you translate that to a physical force, it's going to be in the 
# order of high thousands of tons to 10,000 tons, so the analogy would be the weight of the Eiffel Tower being the kinds of loads it's experiencing,' said Blair Thornton, a professor of 
# marine autonomy, also at the University of Southampton, who has designed and built dozens of robot-operated deep-sea submersibles. Roterman, at the University of Portsmouth, agreed 
# with the Eiffel Tower analogy. His work has taken him to near Titanic depths in the Indian Ocean, descending to observe geothermal hot springs at around 11,400 feet in the Japanese 
# submersible, Shinkai 6500. Unlike Titan, Shinkai is owned and operated by the Japanese government, and has been rigorously testing throughout its 1,500-odd missions. Experts and former 
# passengers have questioned the testing regime and safety record of Titan, which had completed about 20 dives. OceanGate Expeditions, the company that owned and operated Titan, has not 
# responded to those criticisms this week. It did not return NBC News's request for comment. But in promotional information released before the accident, it said the 'state-of-the-art 
# vessel' had been designed in collaboration with NASA and Boeing, undergoing what it described as 'incremental testing' for safety. Like many submersibles, the Shinkai 6500 seats its 
# crew in a titanium sphere, whose manufacture ensures less than 1-millimeter of deviation in its curvature. By contrast, the Titan’s experimental design included a carbon-fiber hull and 
# only the ends of the vessel were made from titanium — a departure from the standard practice. Though he has no direct knowledge of Titan's design or construction, or what contributed 
# to its failure, Thornton, the Southampton professor, said catastrophic incidents can be the result of minute 'defects' in manufacturing. 'And when I say defects, I don't mean any kind 
# of negligent defects — just that when you make things there are small, microscopic defects,' he said, adding that this was the only implosion he was aware of involving a manned 
# submersible. When these defects are exposed to huge forces, they can cause deformations in the structure, leading to higher stresses around those areas, Thornton explained. 'The air 
# inside would compress down to a point, and the forces of the water rushing in and the collapse would be enormous,' he said. 'Structurally,' he added, it would be equivalent of the 
# capsule housing the crew 'suddenly vanishing.' 'This whole process from start to finish is happening in the blink of an eyelid, the click of a finger,' he said. 'I don't think there 
# would have been any suffering or any knowledge of what happened."""