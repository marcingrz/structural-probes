import sys
import os
import glob
import matplotlib.pyplot as plt
import shutil

import yaml

script_file = sys.argv[0]
project_folder = os.path.dirname(os.path.dirname(script_file))
data_folder = sys.argv[1]

latex_template = open(project_folder + "/scripts/templates/dependencies.tex").read()

for folder in os.listdir(data_folder):
    print("Processing " + folder + "...")
    for file in os.listdir(data_folder + "/" + folder):
        if not file.endswith(".yaml"):
            continue

        config_file = data_folder + "/" + folder + "/" + file
        if os.path.exists(config_file + ".finished"):
            continue

        # processing
        return_code = os.system("python " + project_folder + "/structural-probes/run_experiment.py " + config_file)
        if return_code != 0:
            raise RuntimeError("processing " + config_file + " FAILED.")

        open(config_file + ".finished", "w").write("finished.")

    print("Processing " + folder + " done.")


for folder in os.listdir(data_folder):
    folder = data_folder + "/" + folder
    root_accuracies = dict()
    spearman_values = dict()
    spearman_values2 = dict()
    uuas_values = dict()

    results = glob.glob(folder + "/*/*/*.yaml")
    for config_file in results:
        config = yaml.safe_load(open(config_file))
        model_name = config['model']['model_name']
        model_layer = config['model']['model_layer']
        task_name = config['probe']['task_name']
        # print(model_name, model_layer, task_name)
        result_folder = os.path.dirname(config_file)

        if task_name == 'parse-depth':
            root_acc_file = result_folder + "/dev.root_acc"
            root_acc = float(open(root_acc_file).read().split()[0])
            if model_name not in root_accuracies:
                root_accuracies[model_name] = []

            root_accuracies[model_name].append((model_layer, root_acc))

            spearman_file = result_folder + "/dev.spearmanr-5_50-mean"
            spearman = float(open(spearman_file).read())
            if model_name not in spearman_values:
                spearman_values[model_name] = []
            spearman_values[model_name].append((model_layer, spearman))

        if task_name == 'parse-distance':
            root_acc_file = result_folder + "/dev.uuas"
            root_acc = float(open(root_acc_file).read())
            if model_name not in uuas_values:
                uuas_values[model_name] = []

            uuas_values[model_name].append((model_layer, root_acc))

            spearman_file = result_folder + "/dev.spearmanr-5_50-mean"
            spearman = float(open(spearman_file).read())
            if model_name not in spearman_values2:
                spearman_values2[model_name] = []
            spearman_values2[model_name].append((model_layer, spearman))

            tikz_file = open(result_folder + "/dev.tikz").read()
            latex_content = latex_template.replace("%TIKZFILE%", tikz_file)
            open(result_folder + "/dev.latex", "w").write(latex_content)

            # tikz-dependency package file rüberkopieren
            # shutil.copy(project_folder + "/scripts/templates/tikz-dependency.sty", result_folder + "/tikz-dependency.sty")

            return_code = os.system("cd " + result_folder + " && pdflatex dev.latex")
            if return_code != 0:
                raise RuntimeError("pdflatex failed.")


    def plot_models(inputs, name):
        for model_name, plot in inputs.items():
            plot.sort()
            plt.plot([r[0] for r in plot], [r[1] for r in plot], label=model_name)

        plt.legend()
        plt.show()
        plt.savefig(name)
        plt.close()

    os.makedirs(folder + "/plots", exist_ok=True)
    plot_models(root_accuracies, folder + "/plots/parse_depth_root_accuracies.png")
    plot_models(spearman_values, folder + "/plots/parse_depth_spearman_values.png")
    plot_models(spearman_values2, folder + "/plots/parse_distance_spearman_values.png")
    plot_models(spearman_values, folder + "/plots/parse_distance_uuas.png")