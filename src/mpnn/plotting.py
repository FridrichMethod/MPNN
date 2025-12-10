import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import json
from itertools import product
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

# plt.rcParams.update(
#     {
#         "font.family": "Palatino",
#         "mathtext.fontset": "custom",
#         "mathtext.rm": "Palatino",
#         "mathtext.it": "Palatino:italic",
#         "mathtext.bf": "Palatino:bold",
#     }
# )

plt.rcParams.update(
    {
        "font.family": "P052",
        "mathtext.fontset": "custom",
        "mathtext.rm": "P052",
        "mathtext.it": "P052:italic",
        "mathtext.bf": "P052:bold",
    }
)

# Custom color scheme
LIGHT_BLUE = "#8CD9FF"
PURPLE = "#7030A0"
GREEN = "#55A630"
RED = "#e74c3c"
CRAYOLA_BLUE = "#5178D5"
ROBIN_EGG_BLUE = "#45BFC6"
APPLE_GREEN = "#89B537"
ORANGE = "#F8961E"
GOLD = "#C7A020"
PINK = "#FD8BCE"
VIRIDIAN = "#2B8D6E"

CUSTOM_CMAP = mcolors.LinearSegmentedColormap.from_list("custom", [LIGHT_BLUE, PURPLE])
CUSTOM_CMAP.set_bad(color="white", alpha=0)

value_pretty_name_dict = {
    "ar": "Autoregressive",
    "diffusion": "Diffusion",
    "normalized_energy": "Normalized energy",
    "token_value": "Token value",
}

likelihood_to_color_dict = {
    "diffusion": ROBIN_EGG_BLUE,
    "normalized_energy": GOLD,
    "ar": PURPLE,   
    "token_value": PINK,
}

def parse_run(run):
    if key != "none" and key not in run.tags:
        print(f"\033[91mSkipping run {run.name} because it does not have the key {key}\033[0m")
        return None

    if "ignore" in run.tags:
        print(f"\033[91mSkipping run {run.name} because it is ignored\033[0m")
        return None

    if run.state != "finished":
        print(f"\033[91mSkipping run {run.name} because it is not finished\033[0m")
        return None

    run_dict = {}
    run_name = run.name
    run_dict["run_id"] = run_name
    run_json_config = json.loads(run.json_config)
    
    # run_debug = run; import pdb; pdb.set_trace()

    run_dict["parameter_count"] = run_json_config.get("parameter_count", {}).get("value", None)
    run_dict["learning_rate"] = run_json_config.get("optim_lr", {}).get("value", None)
    run_dict["weight_decay"] = run_json_config.get("weight_decay", {}).get("value", None)
    run_dict["num_epochs"] = run_json_config.get("num_epochs", {}).get("value", None)

    run_dict["model_name"] = run_name.split("-lr")[0]
    run_name = run_name.split("-lr")[1]

    run_dict["grad_norm"] = "gn1.0" in run_name

    if "train_loss" not in run.summary or "val_loss" not in run.summary:
        print(f"\033[91mSkipping run {run_name} because it does not have the loss summary\033[0m")
        return None

    run_dict["final_val_loss"] = run.summary["val_loss"]
    run_dict["final_train_loss"] = run.summary["train_loss"]

    run_dict["tags"] = run.tags

    return run_dict


class ScalingLaw:
    def __init__(self):
        self.params = None

    def func(self, x, params):
        pass  # implemented by subclass

    def __str__(self):
        pass  # implemented by subclass

    def asymptote(self):
        pass  # implemented by subclass

    def evaluate(self, x):
        return self.func(x, *self.params)

    def fit(self, x, y, p0=None, bounds=(-np.inf, np.inf)):
        popt, pcov = curve_fit(self.func, x, y, p0=p0, bounds=bounds)
        self.params = popt


class ReciprocalScalingLaw(ScalingLaw):
    def __init__(self, var_name="x"):
        super().__init__()
        self.var_name = var_name

    def func(self, x, A, C):
        return A / x + C

    def __str__(self):
        return f"{self.params[0]:.2f}/{self.var_name} + {self.params[1]:.2f}"

    def asymptote(self):
        return self.params[-1]


class PowerScalingLaw(ScalingLaw):
    def __init__(self, var_name="x"):
        super().__init__()
        self.var_name = var_name

    def func(self, x, A, B, C):
        return A / (x**B) + C

    def __str__(self):
        return f"{self.params[0]:.2f}/{self.var_name}^{self.params[1]:.2f} + {self.params[2]:.2f}"

    def asymptote(self):
        return self.params[-1]


class ChinchillaScalingLaw(ScalingLaw):
    def __init__(self, var1_name="N", var2_name="D"):
        super().__init__()
        self.var1_name = var1_name
        self.var2_name = var2_name

    def func(self, x, A, alpha, B, beta, E):
        n, d = x
        return A / (n**alpha) + B / (d**beta) + E

    def __str__(self):
        return (
            f"{self.params[0]:.2f}/{self.var1_name}^{self.params[1]:.2f} + "
            f"{self.params[2]:.2f}/{self.var2_name}^{self.params[3]:.2f} + "
            f"{self.params[4]:.2f}"
        )

    def asymptote(self):
        return self.params[-1]


def back_out_data(loss, law):
    # loss = A / d^B + C
    # d = (A / (loss - C)) ** (1/B)
    A, B, C = law.params
    return (A / (loss - C)) ** (1 / B)

def plot_full_tradeoff(run_list):
    print("Plotting full tradeoff")
    plt.figure(figsize=(8, 6), dpi=300)

    for likelihood in ["ar", "diffusion", "normalized_energy", "token_value"]:
        runs = [run for run in run_list if run["likelihood"] == likelihood]
        plt.scatter([run["final_train_loss"] for run in runs], [run["final_val_loss"] for run in runs], color=likelihood_to_color_dict[likelihood], label=value_pretty_name_dict[likelihood], alpha=0.5)

    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", alpha=0.5)
    plt.axhline(y=0.5045790990193685, color="black", linestyle="--", label="ln 2 and Entropy")
    plt.axhline(y=np.log(2), color="black", linestyle="--")
    plt.ylim(bottom=0.50, top=0.7)
    plt.xlim(left=0.4, right=0.7)

    plt.legend(loc="lower right")

    plt.xlabel("Train loss")
    plt.ylabel("Val loss")
    plt.title("Full tradeoff")
    plt.savefig("unified/plots/full_tradeoff.png", bbox_inches="tight")

def plot_likelihood_heatmap(run_list, likelihood):
    print(f"Plotting {likelihood} heatmap")
    run_list = [
        run 
        for run in run_list
        if run["likelihood"] == likelihood
        and ("dist-w" in run["model_name"] or "dist-normeng-w" in run["model_name"] or "dist-token_value-w" in run["model_name"])
    ]

    width_depth_loss_dict = {}

    for run in run_list:
        model_name = run["model_name"]
        if "-normeng-" in model_name:
            model_name = model_name.replace("-normeng-", "-")
        if "-token_value-" in model_name:
            model_name = model_name.replace("-token_value-", "-")
        width_divider = float(model_name.split("-")[1][1:])
        depth_divider = float(model_name.split("-")[2][1:])
        loss = run["final_val_loss"]
        if (width_divider, depth_divider) not in width_depth_loss_dict:
            width_depth_loss_dict[(width_divider, depth_divider)] = []
        width_depth_loss_dict[(width_divider, depth_divider)].append(loss)

    unique_width_dividers = sorted(set(width_divider for width_divider, _ in width_depth_loss_dict.keys()), reverse=True)
    unique_depth_dividers = sorted(set(depth_divider for _, depth_divider in width_depth_loss_dict.keys()), reverse=True)

    width_depth_loss_matrix = np.full((len(unique_width_dividers), len(unique_depth_dividers)), np.nan)

    for width_index, depth_index in product(range(len(unique_width_dividers)), range(len(unique_depth_dividers))):
        width_divider = unique_width_dividers[width_index]
        depth_divider = unique_depth_dividers[depth_index]
        if (width_divider, depth_divider) in width_depth_loss_dict:
            width_depth_loss_matrix[width_index, depth_index] = min(width_depth_loss_dict[(width_divider, depth_divider)])

    plt.figure(figsize=(8, 6), dpi=300)
    plt.imshow(width_depth_loss_matrix, cmap=CUSTOM_CMAP)
    plt.colorbar(label="Validation loss")
    plt.xticks(range(len(unique_depth_dividers)), unique_depth_dividers)
    plt.yticks(range(len(unique_width_dividers)), unique_width_dividers)
    plt.xlabel("Depth divider")
    plt.ylabel("Width divider")
    plt.title(f"{value_pretty_name_dict[likelihood]} heatmap")

    for i in range(len(unique_width_dividers)):
        for j in range(len(unique_depth_dividers)):
            if np.ma.is_masked(width_depth_loss_matrix[i, j]):
                data_str = "N/A"
            else:
                data_str = f"{width_depth_loss_matrix[i, j]:.3f}"
            _ = plt.text(j, i, data_str, ha="center", va="center", color="black")

    plt.savefig(f"unified/plots/{likelihood}_heatmap.png", bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--build_cache", action="store_true")
    args = parser.parse_args()
    mode = args.mode

    key, project_name = {
        "binary-ctx16": ("binary-pretraining", "kothasuhas/unified-rlpt"),
    }[mode]

    if args.build_cache:
        run_list = []
        runs = wandb.Api().runs(project_name)
        for run in tqdm(runs):
            run_dict = parse_run(run)
            if run_dict is not None:
                print(run_dict["run_id"])
                run_list.append(run_dict)

        pickle.dump(run_list, open(f"unified/cache/{mode}_run_list.pkl", "wb"))
    else:
        run_list = pickle.load(open(f"unified/cache/{mode}_run_list.pkl", "rb"))

    if mode == "binary-ctx16":
        run_list = [
            run
            for run in run_list
            if "with_proposal" not in run["run_id"]
            and run["final_val_loss"] < 0.7
            and "suffix_len15-prefixes16-3200" in run["tags"]
        ]

        for run in run_list:
            if "mdlm" in run["run_id"]:
                run["likelihood"] = "diffusion"
            elif "normeng" in run["run_id"]:
                run["likelihood"] = 'normalized_energy'
            elif "token_value" in run["run_id"]:
                run["likelihood"] = "token_value"
            else:
                run["likelihood"] = "ar"

        plot_full_tradeoff(run_list)
        plot_likelihood_heatmap(run_list, "ar")
        plot_likelihood_heatmap(run_list, "normalized_energy")
        # plot_likelihood_heatmap(run_list, "diffusion")
    else:
        raise ValueError(f"Invalid mode: {mode}")



## Some reference plotting code for style guidelines

# def plot_200M_sample(
#     losses_for_200M, power_law_200M, run_losses_200M, best_single_model_hparams_ensemble_200M, epoched_data_scaling_law
# ):
#     # Figure 1
#     with plt.rc_context({"font.size": 12}):
#         plt.figure(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)
#         param_counts = [param_str_to_count[model_size] for model_size in losses_for_200M.keys()]
#         min_x = min(param_counts)
#         max_x = max(
#             [max(x_data * param_str_to_count[model_size]) for model_size, (x_data, _, _) in losses_for_200M.items()]
#         )

#         chinchilla_losses = [3.83752, 3.78538, 3.74974, 3.76432]
#         plt.scatter(param_counts, chinchilla_losses, color=BASELINE_COLOR, s=50)
#         plt.plot(param_counts, chinchilla_losses, color=BASELINE_COLOR, label="Epoched recipe")

#         plt.scatter(param_counts, run_losses_200M, color=PURPLE, s=50)
#         model_x_fit = np.linspace(min_x, max_x * 25, 5000)
#         model_y_fit = power_law_200M.evaluate(model_x_fit)
#         plt.plot(model_x_fit, model_y_fit, "--", color=PURPLE, label="Model scaling")

#         # # add a shaded gray region for any y value below asymptote
#         # plt.fill_between(
#         #     model_x_fit,
#         #     3.1,
#         #     power_law_200M.asymptote(),
#         #     color=PURPLE,
#         #     alpha=0.2,
#         #     label="Impossible with standard scaling, infinite compute",
#         # )

#         for model_size, (x_data, y_data, power_law) in losses_for_200M.items():
#             x_fit = np.linspace(min(x_data * param_str_to_count[model_size]), max_x * 25, 5000)
#             y_fit = power_law.evaluate(x_fit / param_str_to_count[model_size])
#             plt.scatter(x_data * param_str_to_count[model_size], y_data, s=50, color=param_str_color_dict[model_size])
#             plt.plot(
#                 x_fit,
#                 y_fit,
#                 "--",
#                 color=param_str_color_dict[model_size],
#                 label=f"{value_pretty_name_dict[model_size]} ensembles",
#             )

#         key_to_pretty_map = {
#             "distill-8ens": "8-Ensemble distill",
#             "self-distill": "Self-distill",
#         }

#         key_to_color_map = {
#             "distill-8ens": ENSEMBLE_COLOR,
#             "self-distill": SELF_DISTILL_COLOR,
#         }

#         # Best 8-mixture distill run: 300m4k-209Mx16-dclm+ens8x0730^0.9-cos-lr0.0030-wd0.10-bs64 (3.3635)
#         # Best self-distill run: 300m4k-209Mx16-dclm+sd0805^0.75-cos-lr0.0030-wd0.10-bs64 (3.43243)
#         plt.scatter(
#             0.3,
#             3.43243,
#             color=key_to_color_map["self-distill"],
#             marker="*",
#             s=120,
#             zorder=6,
#             label=f"{key_to_pretty_map['self-distill']}",
#         )

#         plt.scatter(
#             0.3,
#             3.3635,
#             color=key_to_color_map["distill-8ens"],
#             marker="*",
#             s=120,
#             zorder=6,
#             label=f"{key_to_pretty_map['distill-8ens']}",
#         )

#         plt.legend(loc="upper right")
#         plt.grid(True, which="both", linestyle="--", alpha=0.3)
#         plt.xscale("log")
#         plt.xlabel("Total parameter count")
#         plt.ylabel("DCLM Loss")
#         plt.xlim(right=14.0)
#         plt.title("Validation loss")
#         plt.savefig("experiments/data_efficiency/plots/200M_sample.png", bbox_inches="tight")
#         plt.close()

#     # Figure 2: Comparing parameter count and single model hparams ensemble
#     plt.figure(figsize=(6, 4), dpi=300)
#     xmax_multiplier = 1
#     model_x_fit = np.linspace(min_x, max_x * xmax_multiplier, 5000)
#     model_y_fit = power_law_200M.evaluate(model_x_fit)
#     plt.scatter(param_counts, run_losses_200M, color=REGULARIZED_COLOR, s=50, zorder=5)
#     plt.plot(
#         model_x_fit,
#         model_y_fit,
#         "--",
#         color=REGULARIZED_COLOR,
#         label=f"Model scaling: (Fit: {power_law_200M})",
#         zorder=4,
#         alpha=0.8,
#     )

#     for model_size, (x_data, y_data, power_law) in [("300m4k", best_single_model_hparams_ensemble_200M[1:])]:
#         x_fit = np.linspace(min(x_data * param_str_to_count[model_size]), max_x * xmax_multiplier, 5000)
#         y_fit = power_law.evaluate(x_fit / param_str_to_count[model_size])
#         plt.scatter(
#             x_data * param_str_to_count[model_size], y_data, s=50, color=param_str_color_dict[model_size], zorder=5
#         )
#         print(x_data, y_data)
#         plt.plot(
#             x_fit,
#             y_fit,
#             "--",
#             color=param_str_color_dict[model_size],
#             label=f"{value_pretty_name_dict[model_size]} ensembles (Fit: {power_law})",
#             zorder=4,
#             alpha=0.8,
#         )

#     plt.legend()
#     # plt.grid(True, alpha=0.3)
#     plt.xscale("log")
#     plt.xticks(param_counts, ["150M", "300M", "600M", "1.4B"])
#     plt.xticks([], [], minor=True)
#     plt.xlabel("Total parameter count")
#     plt.ylabel("Loss")
#     plt.title("Ensemble member scaling")
#     plt.grid(True, which="both", linestyle="--", alpha=0.3)
#     plt.savefig("experiments/data_efficiency/plots/200M_ensemble_vs_parameter_scaling.png", bbox_inches="tight")
#     plt.close()

#     # Figure 3 (new figure 1 (8/22))
#     fig, ax1 = plt.subplots(figsize=(6, 5), dpi=300)
#     xmax_multiplier = 5
#     chinchilla_losses = [3.83752, 3.78538, 3.74974, 3.76432]
#     ax1.scatter(param_counts, chinchilla_losses, color=BASELINE_COLOR, s=50)
#     ax1.plot(param_counts, chinchilla_losses, color=BASELINE_COLOR, label="Standard recipe")

#     ax1.scatter(param_counts, run_losses_200M, color=REGULARIZED_COLOR, s=50, zorder=6)
#     model_x_fit = np.linspace(min(x_data * param_str_to_count["150m4k"]), max_x * xmax_multiplier, 5000)
#     model_y_fit = power_law_200M.evaluate(model_x_fit)
#     ax1.plot(
#         model_x_fit,
#         model_y_fit,
#         "--",
#         color=REGULARIZED_COLOR,
#         label=f"Regularized recipe\n(Fit: {power_law_200M})",
#         zorder=4,
#         alpha=0.8,
#     )

#     for model_size, (x_data, y_data, power_law) in [("300m4k", best_single_model_hparams_ensemble_200M[1:])]:
#         x_fit = np.linspace(min(x_data * param_str_to_count[model_size]), max_x * xmax_multiplier, 5000)
#         y_fit = power_law.evaluate(x_fit / param_str_to_count[model_size])
#         ax1.scatter(
#             x_data * param_str_to_count[model_size], y_data, s=50, color=param_str_color_dict[model_size], zorder=5
#         )
#         print(x_data, y_data)
#         ax1.plot(
#             x_fit,
#             y_fit,
#             "--",
#             color=param_str_color_dict[model_size],
#             label=f"Ensembling recipe\n(Fit: {power_law})",
#             zorder=4,
#             alpha=0.8,
#         )

#     infinite_compute_asymptote = 3.17

#     ax1.axhline(y=min(chinchilla_losses), color=BASELINE_COLOR, linestyle=":", zorder=3, alpha=0.5)
#     ax1.axhline(y=power_law_200M.asymptote(), color=REGULARIZED_COLOR, linestyle=":", zorder=3, alpha=0.5)
#     ax1.axhline(y=power_law.asymptote(), color=param_str_color_dict[model_size], linestyle=":", zorder=3, alpha=0.5)
#     ax1.axhline(
#         y=infinite_compute_asymptote,
#         linestyle=":",
#         label="Joint scaling recipe\nasymptote ($N,K\\to\\infty$)",
#         color=TIERED_COLOR,
#         zorder=3,
#     )

#     def data_efficiency_from_loss(loss):
#         return back_out_data(loss, epoched_data_scaling_law) / back_out_data(
#             min(chinchilla_losses), epoched_data_scaling_law
#         )

#     def loss_from_data_efficiency(data_efficiency):
#         effective_data = data_efficiency * back_out_data(min(chinchilla_losses), epoched_data_scaling_law)
#         return epoched_data_scaling_law.evaluate(effective_data)

#     plt.scatter(
#         0.3,
#         3.3635,
#         color=ENSEMBLE_COLOR,
#         # edgecolors="black",
#         linewidths=0.8,
#         marker="*",
#         s=150,
#         zorder=6,
#         label="8-ensemble distill: 3.36",
#     )

#     plt.scatter(
#         0.3,
#         3.43243,
#         color=SELF_DISTILL_COLOR,
#         # edgecolors="black",
#         linewidths=0.8,
#         marker="*",
#         s=150,
#         zorder=6,
#         label="Self-distill: 3.43",
#     )

#     ax1.legend(framealpha=1.0)

#     handles, labels = plt.gca().get_legend_handles_labels()
#     # bottom left legend
#     ax1.legend(handles=handles[-2:], labels=labels[-2:], framealpha=1.0, loc="lower left")

#     ax1.set_xscale("log")
#     ax1.set_xticks(param_counts)
#     ax1.set_xticklabels(["150M", "300M", "600M", "1.4B"])
#     ax1.tick_params(axis="x", which="minor", bottom=False)
#     ax1.set_xlim(right=2 * xmax_multiplier)
#     ax1.set_xlabel("Total parameter count")
#     ax1.set_ylabel("DCLM validation loss")
#     ax1.set_title("Comparing scaling recipes with no compute constraints")

#     # Set the secondary axis limits using the formula 1/loss + 5
#     secax = ax1.secondary_yaxis("right", functions=(data_efficiency_from_loss, loss_from_data_efficiency))
#     secax.set_ylabel("Data efficiency")
#     important_losses = [
#         min(chinchilla_losses),
#         power_law_200M.asymptote(),
#         power_law.asymptote(),
#         infinite_compute_asymptote,
#     ]
#     colors = [BASELINE_COLOR, REGULARIZED_COLOR, param_str_color_dict[model_size], TIERED_COLOR]
#     secax.set_yticks(
#         [data_efficiency_from_loss(loss) for loss in important_losses],
#         [f"${data_efficiency_from_loss(loss):.2f}\\times$" for loss in important_losses],
#     )

#     for tick, color in zip(secax.yaxis.get_ticklabels(), colors, strict=False):
#         tick.set_color(color)

#     plt.savefig("experiments/data_efficiency/plots/figure_1_8_22.png", bbox_inches="tight")
#     plt.close()

#     # Figure 3: Marginal returns
#     plt.figure(figsize=(8, 5), dpi=300)
#     # plot AB/(A+Ce^Bx) for the same x values
#     x_fit = np.logspace(np.log10(0.00001), np.log10(max_x * 25), 5000)
#     A, B, C = power_law_200M.params
#     plt.plot(
#         x_fit,
#         [A * B / (A + C * np.exp(B * np.log(x))) for x in x_fit],
#         "--",
#         color=PURPLE,
#         label=f"Model scaling: (Fit: {A:.2f}/x^{B:.2f} + {C:.2f})",
#     )
#     for model_size, (x_data, _, power_law) in losses_for_200M.items():
#         A, B, C = power_law.params
#         x_fit = np.logspace(np.log10(min(x_data * param_str_to_count[model_size]) * 0.0001), np.log10(max_x * 25), 5000)
#         y_fit = A * B / (A + C * np.exp(B * np.log(x_fit / param_str_to_count[model_size])))
#         plt.plot(
#             x_fit,
#             y_fit,
#             "--",
#             label=f"{value_pretty_name_dict[model_size]} ensembles (Fit: {A:.2f}/(x^{B:.2f}) + {C:.2f})",
#         )
#     plt.legend()
#     plt.xscale("log")
#     plt.xlabel("Total model parameters (billions)")
#     plt.ylabel("Returns on excess log loss")
#     plt.title("Returns on excess log loss for 200M tokens")
#     plt.savefig("experiments/data_efficiency/plots/200M_sample_AB_A_Ce_Bx.png", bbox_inches="tight")
#     plt.close()