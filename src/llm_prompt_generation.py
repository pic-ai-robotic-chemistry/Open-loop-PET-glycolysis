import numpy as np
import pandas as pd
import os


BASE = r'C:\Users\darkn\PycharmProjects\AcidLLMRecommendation\data\ustc_data\数据整理'


def generate_prompt_part1():
    prompt = ("Polyethylene terephthalate (PET) is a widely used thermoplastic known for its strength, stability, "
              "and safety, but its increasing demand has led to significant environmental challenges and reliance on "
              "non-renewable resources. Efficient recycling of PET, particularly through ethylene glycol (EG) "
              "glycolysis to produce bis(hydroxyethyl) terephthalate (BHET), offers a sustainable solution. This "
              "process, operating under mild conditions, has been commercialized, with catalysts playing a key role "
              "in enhancing reaction efficiency. This study focuses on developing high-performance Lewis acid-base "
              "catalysts guided by machine learning and establishing an automated platform for plastic degradation.")
    prompt += ("Here we already have tested several pairs of acids and alkaline bases. All the acids and alkaline "
               "bases as well as their corresponding catalytic yields are listed below:\n\n")
    bo = pd.read_excel(os.path.join(BASE, 'bo_data.xlsx'))
    baseline = pd.read_excel(os.path.join(BASE, 'human_data.xlsx'))

    for i in range(len(bo)):
        row = bo.iloc[i]
        prompt += f"Acid: {row['acid']}, alkaline base: {row['base']}, catalytic yield: {row['产率']}\n"

    for i in range(len(baseline)):
        row = baseline.iloc[i]
        prompt += f"Acid: {row['acid']}, alkaline base: {row['base']}, catalytic yield: {row['产率']}\n"

    return prompt


def generate_prompt_part2():
    prompt = "Here are some extra chemical information:\n\n"
    prompt += "1. The Role of Lewis Acids in Catalyzing Plastic Degradation\n"
    prompt += "1.1 Effects on Anions\n"
    prompt += ("The moderate basicity of the counter anion in metal salts yields optimal results. Anions serve two key "
               "roles: coordination with metal ions and acting as bases. The binding constant between anions and "
               "metal ions should remain at an intermediate level, balancing two effects:\n")
    prompt += ("(1) Reducing anion-metal ion binding, which facilitates the coordination of metal ions with ester "
               "carbonyl groups. This coordination activates the carbonyl group, enhancing the nucleophilic attack by "
               "the oxygen atom of the alcohol hydroxyl group.\n")
    prompt += ("(2) Exhibiting strong anionic basicity, which accelerates the deprotonation of the alcohol hydroxyl "
               "group.\n")
    prompt += ("The ability of metal ions to bind with counter anions and the basicity of the anions are inherently in "
               "conflict; therefore, an optimal intermediate value must be determined.\n")
    prompt += "1.2 Effects of Metal Ions\n"
    prompt += ("Metal ions coordinate with the oxygen atom of the carbonyl group, facilitating the degradation process "
               "through two key mechanisms:\n")
    prompt += "(1) Increasing the solubility of polymers in ethylene glycol, thereby promoting the reaction.\n"
    prompt += ("(2) Activating the carbonyl group by enhancing the electrophilicity of the carbonyl carbon, making it "
               "more susceptible to nucleophilic attack by ethylene glycol.\n")
    prompt += ("Current experimental results indicate that zinc ions demonstrate the most effective activation of "
               "carbonyl groups.\n")
    return prompt


def generate_prompt_part3():
    prompt = ("Now we need to consider the synergistic effects of Lewis acid-base pairs in the catalytic degradation "
              "of plastics. Assuming that the degradation yield of the Lewis acid alone is x, and the yield of the "
              "base alone is y, if the combined yield significantly exceeds the larger one in x and y, the pair is "
              "considered to exhibit positive synergy. Conversely, if the combined yield is significantly less than "
              "the smaller one of x and y, they are considered to exhibit negative synergy. \n")
    prompt += ("Our goal is to identify the shared characteristics of acids or bases that exhibit positive synergy. "
               "For example, if acid A and base B demonstrate positive synergistic effects, we aim to determine the "
               "common characteristics that acid C might share with acid A to predict that it could also exhibit "
               "positive synergy with base B. By leveraging such characteristics, we can systematically recommend "
               "Lewis acid-base pairs with positive synergistic effects. \n")
    prompt += ("\n\n Please analyze the data and generate 5 hypotheses about the preferences and data trend based on "
               "the above Lewis acid and base theory for further research. Please also provide your logic as well as "
               "the data points that suppport your hypotheses. \n\n")
    return prompt


def generate_prompt():
    prompt = generate_prompt_part1() + generate_prompt_part2() + generate_prompt_part3()
    return prompt


def generate_suggestion_prompt():
    avail_acid_info = pd.read_excel('data/ustc_data/acid.xlsx')['name'].tolist()
    avail_base_info = pd.read_excel('data/ustc_data/base.xlsx')['name'].tolist()

    prompt = ("For each one of the above hypotheses, please recommend 3 pairs of acid and base for higher catalytic "
              "yield. Please do not recommend any acid-base pair that is already tested and given above.\n")
    prompt += ("All available acids and alkaline bases are listed below. Please only recommend the acids and alkaline "
               "bases that are avaiable in the database.\n")

    prompt += "The available acids are:\n"
    for i in range(len(avail_acid_info)):
        prompt += f"{i + 1}. {avail_acid_info[i]}\n"
    prompt += "\n\nThe available alkaline bases are: \n\n"
    for i in range(len(avail_base_info)):
        prompt += f"{i + 1}. {avail_base_info[i]}\n"

    prompt += ("\nPlease give out each of the above hypotheses first, then the 3 suggestions corresponding to the "
               "hypothesis.")
    return prompt


if __name__ == '__main__':
    prompt = generate_prompt()
    print(prompt)

    