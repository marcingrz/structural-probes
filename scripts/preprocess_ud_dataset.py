import os
import sys

# configs für das skript:
MODELS = [("bert-large-cased",24), ("bert-base-cased",12), ("bert-base-multilingual-cased",12)]

# input: ordner von universal dependencies
script_file = sys.argv[0]
input = sys.argv[1]

# dev, test, train conllu datei suchen
dev_file = None
test_file = None
train_file = None

for file in os.listdir(input):
    if file.endswith("dev.conllu"):
        dev_file = file
    elif file.endswith("test.conllu"):
        test_file = file
    elif file.endswith("train.conllu"):
        train_file = file

if dev_file is None:
    raise RuntimeError("dev file not found.")
if test_file is None:
    raise RuntimeError("test file not found.")
if train_file is None:
    raise RuntimeError("train file not found.")

script_folder = os.path.dirname(script_file)
output_folder = os.path.dirname(script_folder) + "/data/" + os.path.basename(input)
os.makedirs(output_folder, exist_ok=True)

def generate_conllx_path(input_file):
    return input_file.replace(".conllu", ".conllx")

input_files = [dev_file, test_file, train_file]

for input_file in input_files:
    # convert conllu zu conllx, in den results ordner
    conllx_file = output_folder + "/" + generate_conllx_path(input_file)
    return_code = os.system("perl " + script_folder + "/conllu_to_conllx.pl < " + input + "/" + input_file + " > " + conllx_file)
    if return_code != 0:
        raise RuntimeError("conllu_to_conllx failed.")

    # convert conllu zu raw ausführen
    raw_txt_file = conllx_file + ".txt"
    return_code = os.system(
        "python " + script_folder + "/convert_conll_to_raw.py " + conllx_file + " > " + raw_txt_file)
    if return_code != 0:
        raise RuntimeError("convert_conll_to_raw failed.")

    # convert raw to bert ausführen (hier auf die modell einstellung achten)
    for (model,layers) in MODELS:
        bert_file = raw_txt_file.replace(".txt", "." + model + ".hdf5")
        return_code = os.system(
            "python " + script_folder + "/convert_raw_to_bert.py " + raw_txt_file + " " + bert_file + " " + model)
        if return_code != 0:
            raise RuntimeError("convert_raw_to_bert failed.")

# 2*X config dateien generieren:
# parse distance test mit gewähltem modell (z.b. bert base), mit layer x
# parse depth test mit gewählten modell (z.b. bert base), mit layer x
for template_file in os.listdir(script_folder + "/templates"):
    template_data = open(script_folder + "/templates/" + template_file).read()
    template_data = template_data.replace("%ROOT%", output_folder)
    template_data = template_data.replace("%TRAIN_CONLLX%", generate_conllx_path(train_file))
    template_data = template_data.replace("%TEST_CONLLX%", generate_conllx_path(test_file))
    template_data = template_data.replace("%DEV_CONLLX%", generate_conllx_path(dev_file))

    for (model, layers) in MODELS:
        for layer in range(layers):
            config = template_data
            config = config.replace("%MODEL%", model)
            config = config.replace("%LAYER%", str(layer))

            config_path = output_folder + "/" + template_file.replace("template", model + "_" + str(layer))
            open(config_path, "w").write(config)

