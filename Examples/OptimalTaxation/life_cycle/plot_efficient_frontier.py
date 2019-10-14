import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools

def plot_frontier_carlos(pv_rev, pv_util, tk, tl, tc, conf, file_name=""):
    # create figure with 2x2 subplots
    cols = 3
    rows = 2

    plt.rc('text', usetex=True)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

    data_list = [np.array(tk)*100, np.array(tl)*100, np.array(tc)*100]
    title_list = ["Capital Tax", "Labor Income Tax", "Consumption Tax"]

    # Plot taxes
    for data, title, idx in zip(data_list, title_list,
                                itertools.product([1], range(cols))):
        axes[idx[0], idx[1]].scatter(pv_rev, data, 1)
        axes[idx[0], idx[1]].set_ylim([-5, 100])
        axes[idx[0], idx[1]].set_title(title + " vs. Revenue")

    axes[0, 0].scatter(pv_rev, pv_util, 0.5)
    axes[0, 0].set_title("Utility vs. Revenue")

    idx_sorted = np.argsort(pv_rev)
    larger_z = np.diff(np.array(pv_rev)[idx_sorted]) > 0
    axes[0, 1].scatter(np.array(pv_rev)[idx_sorted][1:], np.diff(np.array(pv_util)[idx_sorted])/np.diff(np.array(pv_rev)[idx_sorted]), 2)
    axes[0, 1].set_ylim([-10, 0])
    axes[0, 1].set_title("MEB")

    axes[0, 2].set_xticklabels([])
    axes[0, 2].set_yticklabels([])
    axes[0, 2].axis('off')
    flag_borrowing = conf["borrowing"] == 0
    if (flag_borrowing):
        borrow = "no"
    else:
        borrow = "yes"

    axes[0, 2].text(0.2, 0.3, "$\\\\ \gamma = $ " + str(conf["gamma"]) + "\\\\" +
                              "$\eta = $ " + str(conf["eta"]) +
                              "\\\\ Borrowing : " + borrow +
                              "\\\\ Tolerance : " + str(conf["tol"]), fontsize=25)
    plt.savefig(file_name + '_Pareto_Frontier.pdf')



def plot_frontier(pv_rev, pv_util, tk, tl, tc, file_name=""):
    # create figure with 2x2 subplots
    plt.figure(2)

#    plt.subplot(411)
    t = plt.scatter(pv_rev, pv_util, 2, c=tk)
    plt.colorbar(t)
    plt.title("Capital tax")
    plt.savefig(file_name + '_capital_tax.pdf')

#
#     plt.subplot(412)
    plt.figure(3)
    t = plt.scatter(pv_rev, pv_util, 2, c=tl)
    plt.colorbar(t)
    plt.title("Labour tax")
    plt.savefig(file_name + '_labour_tax.pdf')

    plt.figure(4)
    t = plt.scatter(pv_rev, pv_util, 2, c=tc)
    plt.colorbar(t)
    plt.title("Consumption tax")
    plt.savefig(file_name + '_consumption_tax.pdf')

#     plt.subplot(413)

    plt.figure(5)

    plt.scatter(pv_rev, tl, 2)
    plt.title('Labour tax vs. total revenue')
    plt.savefig(file_name + '_labour_tax_revenue.pdf')

#
#     plt.subplot(414)
    plt.figure(6)

    idx_sorted = np.argsort(pv_rev)
    larger_z = np.diff(np.array(pv_rev)[idx_sorted]) > 0
    plt.plot(np.array(pv_rev)[idx_sorted][1:][larger_z], np.diff(np.array(pv_util)[idx_sorted])[larger_z]/np.diff(np.array(pv_rev)[idx_sorted])[larger_z], ".")
    plt.title('MEB')
    plt.savefig(file_name + '_MEB.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    data = np.loadtxt(args.filename, delimiter=",")
    data = data.T
    [pv_rev, pv_util, tk, tl, tc] = data

    plot_frontier(pv_rev, pv_util, tk, tl, tc, file_name=str.split(args.filename, "_res")[0])