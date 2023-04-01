import matplotlib.pyplot as plt

# data for Algo-FDR line
fdr_tau = [0.7546191562630375, 0.7545965485014585, 0.7583210139276974, 
           0.7567413986597848, 0.7567166870354417, 0.7566712052023732,
           0.7506231251800178, 0.7558203905208568, 0.7568707523548349, 
           0.7507507794941347]
accuracy = [0.5927321200880955, 0.5925002897878753, 0.591283180711719,
            0.591283180711719, 0.5915729685869944, 0.5914570534368842, 
            0.5940651443143619, 0.5914570534368842, 0.5912252231366639, 
            0.5955140836907384]

# data for Algo-SR line
fdr_tau2 = [0.7610499868882136, 0.761069978555769, 0.7606221724857138, 
            0.7595097736516832, 0.7592881963405731, 0.7592881963405731, 
            0.7592881963405731, 0.7592881963405731, 0.7592881963405731, 
            0.7592881963405731]
accuracy2 = [0.5928480352382056, 0.5928480352382056, 0.5926741625130405, 
             0.5928480352382056, 0.5929639503883157, 0.5929639503883157, 
             0.5929639503883157, 0.5929639503883157, 0.5929639503883157, 
             0.5929639503883157]

# plot the lines
plt.scatter(fdr_tau, accuracy, color='red', label='Algo-FDR')
plt.scatter(fdr_tau2, accuracy2, color='blue', label='Algo-SR')

# set plot title and axis labels
plt.title('Accuracy vs. gamma_FDR')
plt.xlabel('gamma_FDR')
plt.ylabel('Accuracy')

# add legend
plt.legend()

# show the plot
plt.savefig("plot_accuracy_vs_gamma_fdr.png")