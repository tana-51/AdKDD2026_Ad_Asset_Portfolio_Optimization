import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot(
        result_df, 
        varying_list, 
        varying_name, 
        axis_name, 
        save_path,
        log_scale=False
):

    legend = ["Logging Policy", "IPS-PG", "Reg-based", "PI-PG", "Independent Reg-based","Ours", rf"Ours (without $\lambda_g$)"]
    palette = ["black", "tab:red", "tab:blue", "tab:purple", "tab:green", "tab:orange", "tab:gray"]

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10,6), tight_layout=True)

    ax = fig.add_subplot(1,1,1)
    sns.lineplot(
            linewidth=4,
            x=varying_name,
            y="value",
            hue="method",
            ax=ax,
            legend=False,
            data=result_df,
            markers=True,
            dashes=False,
            markersize=15,
            marker="o",
            palette=palette,
            errorbar=None,
    )
    sns.lineplot(
            linewidth=4,
            x=varying_name,
            y="value",
            hue="method",
            ax=ax,
            legend=False,
            data=result_df,
            markers=True,
            dashes=False,
            markersize=15,
            marker="o",
            palette=palette,
    )
    ax.set_ylabel("Policy Value", fontsize = 20)
    ax.tick_params(axis="y", labelsize=15)
    ax.yaxis.set_label_coords(-0.07, 0.5)
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(axis_name, fontsize = 20)
    ax.set_xticks(varying_list)
    ax.set_xticklabels(varying_list, fontsize=15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.legend(legend)
    fig.savefig(save_path+f"{varying_name}_value.png")


    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10,6), tight_layout=True)

    ax = fig.add_subplot(1,1,1)
    sns.lineplot(
            linewidth=4,
            x=varying_name,
            y="rel_value",
            hue="method",
            ax=ax,
            legend=False,
            data=result_df,
            markers=True,
            dashes=False,
            markersize=15,
            marker="o",
            palette=palette,
            errorbar=None,
    )
    sns.lineplot(
            linewidth=4,
            x=varying_name,
            y="rel_value",
            hue="method",
            ax=ax,
            legend=False,
            data=result_df,
            markers=True,
            dashes=False,
            markersize=15,
            marker="o",
            palette=palette,
    )
    ax.set_ylabel("Relative Policy Value", fontsize = 20)
    ax.tick_params(axis="y", labelsize=15)
    ax.yaxis.set_label_coords(-0.07, 0.5)
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(axis_name, fontsize = 20)
    ax.set_xticks(varying_list)
    ax.set_xticklabels(varying_list, fontsize=15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.legend(legend)
    fig.savefig(save_path+f"{varying_name}_rel_value.png")


    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10,6), tight_layout=True)

    ax = fig.add_subplot(1,1,1)
    sns.lineplot(
            linewidth=4,
            x=varying_name,
            y="rel_value_max",
            hue="method",
            ax=ax,
            legend=False,
            data=result_df,
            markers=True,
            dashes=False,
            markersize=15,
            marker="o",
            palette=palette,
            errorbar=None,
    )
    sns.lineplot(
            linewidth=4,
            x=varying_name,
            y="rel_value_max",
            hue="method",
            ax=ax,
            legend=False,
            data=result_df,
            markers=True,
            dashes=False,
            markersize=15,
            marker="o",
            palette=palette,
    )
    max_ = "max"
    ax.set_ylabel(rf"Relative Policy Value ($V{max_}$)", fontsize = 20)
    ax.tick_params(axis="y", labelsize=15)
    ax.yaxis.set_label_coords(-0.07, 0.5)
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(axis_name, fontsize = 20)
    ax.set_xticks(varying_list)
    ax.set_xticklabels(varying_list, fontsize=15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.legend(legend)
    fig.savefig(save_path+f"{varying_name}_rel_value_max.png")
        