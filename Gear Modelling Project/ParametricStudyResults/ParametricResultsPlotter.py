import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def main():

    a_cpress = []
    a_mises = []
    e_cpress = []
    e_mises = []
    z_cpress = []
    z_mises = []
    r_cpress = []
    r_mises = []

    a_arr = []
    e_arr = []
    z_arr = []
    r_arr = []

    for directory in os.listdir():

        if directory == "CentreDistance":
            for file in os.listdir("CentreDistance"):
                centre_distance = file.split("_")[2] + "." + file.split("_")[3]

                a_arr += [float(centre_distance)]

                if file.split(".")[-1] == 'csv':
                    if file.split("_")[-1].split(".")[0] == 'CPRESS':
                        df = pd.read_csv("CentreDistance//" + file)
                        df.drop([0], inplace=True)

                        array = df.values[:, 1:].astype('float')

                        cpress = np.amax(array, axis=0)

                        n = len(cpress)
                        max_cpress = np.max(cpress[int(9/16*n):int(13/16*n)])

                        a_cpress += [float(max_cpress/1000/30)]

                    if file.split("_")[-1].split(".")[0] == 'MISES':
                        df = pd.read_csv("CentreDistance//" + file)
                        df.drop([0], inplace=True)

                        array = df.values[:, 1:].astype('float')

                        mises = np.amax(array, axis=0)

                        max_mises = max(mises[:int(0.9 * len(mises))])

                        a_mises += [float(max_mises)]

        if directory == "ElasticModulus":
            for file in os.listdir("ElasticModulus"):
                modulus = file.split("_")[2] + "." + file.split("_")[3]

                e_arr += [float(modulus)]

                if file.split(".")[-1] == 'csv':
                    if file.split("_")[-1].split(".")[0] == 'CPRESS':
                        df = pd.read_csv("ElasticModulus//" + file)
                        df.drop([0], inplace=True)

                        array = df.values[:, 1:].astype('float')

                        cpress = np.amax(array, axis=0)

                        n = len(cpress)
                        max_cpress = np.max(cpress[int(9/16*n):int(13/16*n)])

                        e_cpress += [float(max_cpress/1000/30)]

                    if file.split("_")[-1].split(".")[0] == 'MISES':
                        df = pd.read_csv("ElasticModulus//" + file)
                        df.drop([0], inplace=True)
                        roll_angle = df.columns[1:].astype('float')
                        array = df.values[:, 1:].astype('float')

                        mises = np.amax(array, axis=0)

                        red = abs(255/930 * float(modulus) - 70*255/930)
                        blue = abs(-255/930 * float(modulus) + 8500/31)

                        print(modulus)

                        plt.plot(roll_angle, mises, c=(blue/255, 0, red/255, 0.5), label=modulus)

                        max_mises = max(mises[:int(0.9 * len(mises))])
                        max_roll = roll_angle[np.argmax(mises[:int(0.9 * len(mises))])]

                        plt.scatter(max_roll, max_mises, c=(blue/255, 0, red/255), marker='x', s=150)
                        e_mises += [float(max_mises)]

        if directory == "PinionTeeth":
            for file in os.listdir("PinionTeeth"):
                teeth = file.split("_")[2] + "." + file.split("_")[3]

                z_arr += [float(teeth)]

                if file.split(".")[-1] == 'csv':
                    if file.split("_")[-1].split(".")[0] == 'CPRESS':
                        df = pd.read_csv("PinionTeeth//" + file)
                        df.drop([0], inplace=True)

                        array = df.values[:, 1:].astype('float')

                        cpress = np.amax(array, axis=0)

                        n = len(cpress)
                        max_cpress = np.max(cpress[int(9/16*n):int(13/16*n)])

                        z_cpress += [float(max_cpress/1000/30)]

                    if file.split("_")[-1].split(".")[0] == 'MISES':
                        df = pd.read_csv("PinionTeeth//" + file)
                        df.drop([0], inplace=True)

                        array = df.values[:, 1:].astype('float')

                        mises = np.amax(array, axis=0)

                        max_mises = max(mises[:int(0.9 * len(mises))])

                        z_mises += [float(max_mises)]

        if directory == "RootFilletRadius":
            for file in os.listdir("RootFilletRadius"):
                radius = file.split("_")[2] + "." + file.split("_")[3]

                r_arr += [float(radius)]

                if file.split(".")[-1] == 'csv':
                    if file.split("_")[-1].split(".")[0] == 'CPRESS':
                        df = pd.read_csv("RootFilletRadius//" + file)
                        df.drop([0], inplace=True)

                        array = df.values[:, 1:].astype('float')

                        cpress = np.amax(array, axis=0)

                        n = len(cpress)
                        max_cpress = np.max(cpress[int(9/16*n):int(13/16*n)])

                        r_cpress += [float(max_cpress/1000/30)]

                    if file.split("_")[-1].split(".")[0] == 'MISES':
                        df = pd.read_csv("RootFilletRadius//" + file)
                        df.drop([0], inplace=True)

                        array = df.values[:, 1:].astype('float')

                        mises = np.amax(array, axis=0)

                        max_mises = max(mises[:int(0.9 * len(mises))])

                        r_mises += [float(max_mises)]

    plt.title("Maximum Root Bending Stress vs Roll Angle")
    plt.ylabel("Maximum Root Bending Stress [MPa]")
    plt.xlabel("Roll Angle [deg]")
    plt.xlim([8, 12])
    plt.show()

    # Removing duplicated variable values (as each run has two, one for MISES and one for CPRESS
    a_arr = list(dict.fromkeys(a_arr))
    e_arr = list(dict.fromkeys(e_arr))
    z_arr = list(dict.fromkeys(z_arr))
    r_arr = list(dict.fromkeys(r_arr))

    # Converting everything to nparray for easier sorting to ensure plots are in correct order
    a_arr = np.array(a_arr)
    e_arr = np.array(e_arr)
    z_arr = np.array(z_arr)
    r_arr = np.array(r_arr)

    a_mises = np.array(a_mises)
    e_mises = np.array(e_mises)
    z_mises = np.array(z_mises)
    r_mises = np.array(r_mises)
    a_cpress = np.array(a_cpress)
    e_cpress = np.array(e_cpress)
    z_cpress = np.array(z_cpress)
    r_cpress = np.array(r_cpress)

    # Sorting by increasing x (variable) order, so that plots show correctly
    # This is because file order may not be sorted in terms of increasing variable size

    a_idx = a_arr.argsort()
    a_arr, a_mises, a_cpress = a_arr[a_idx], a_mises[a_idx], a_cpress[a_idx]
    e_idx = e_arr.argsort()
    e_arr, e_mises, e_cpress = e_arr[e_idx], e_mises[e_idx], e_cpress[e_idx]
    z_idx = z_arr.argsort()
    z_arr, z_mises, z_cpress = z_arr[z_idx], z_mises[z_idx], z_cpress[z_idx]
    r_idx = r_arr.argsort()
    r_arr, r_mises, r_cpress = r_arr[r_idx], r_mises[r_idx], r_cpress[r_idx]

    print(e_arr, e_mises, e_cpress)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    ax1 = ax[0, 0].twinx()
    ax2 = ax[0, 1].twinx()
    ax3 = ax[1, 0].twinx()
    ax4 = ax[1, 1].twinx()

    ax[0, 0].set_title("Stress vs Centre Distance")
    ax[0, 0].set_xlabel("Centre Distance [mm]")
    ax[0, 0].set_ylabel("Contact Pressure [GPa]", c='r')
    ax[0, 0].plot(a_arr, a_cpress, label="CPRESS", c='r')
    ax[0, 0].legend(loc="upper right")
    ax1.plot(a_arr, a_mises, label="MISES", c='b')
    ax1.set_ylabel("Root Fillet Stress [MPa]", c='b')
    ax1.legend(loc="upper left")

    ax[0, 1].set_title("Stress vs Elastic Modulus")
    ax[0, 1].set_xlabel("Elastic Modulus [GPa]")
    ax[0, 1].set_ylabel("Contact Pressure [GPa]", c='r')
    ax[0, 1].plot(e_arr, e_cpress, label="CPRESS", c='r')
    ax[0, 1].legend(loc="upper right")
    ax2.plot(e_arr, e_mises, label="MISES", c='b')
    ax2.set_ylabel("Root Fillet Stress [MPa]", c='b')
    ax2.legend(loc="upper left")

    ax[1, 0].set_title("Stress vs Pinion Teeth")
    ax[1, 0].set_xlabel("Pinion Teeth [-]")
    ax[1, 0].set_ylabel("Contact Pressure [GPa]", c='r')
    ax[1, 0].plot(z_arr, z_cpress, label="CPRESS", c='r')
    ax[1, 0].legend(loc="upper right")
    ax3.plot(z_arr, z_mises, label="MISES", c='b')
    ax3.set_ylabel("Root Fillet Stress [MPa]", c='b')
    ax3.legend(loc="upper left")

    ax[1, 1].set_title("Stress vs Root Fillet Radius")
    ax[1, 1].set_xlabel("Root Fillet Radius [mm]")
    ax[1, 1].set_ylabel("Contact Pressure [GPa]", c='r')
    ax[1, 1].plot(r_arr, r_cpress, label="CPRESS", c='r')
    ax[1, 1].legend(loc="upper right")
    ax4.plot(r_arr, r_mises, label="MISES", c='b')
    ax4.set_ylabel("Root Fillet Stress [MPa]", c='b')
    ax4.legend(loc="upper left")

    axis_comparison = True

    if axis_comparison:
        ax[0, 0].set_ylim([1.2, 1.4])
        ax[1, 1].set_ylim([1.2, 1.4])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
