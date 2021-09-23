import numpy as np
import IPython
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable



def plot_grid(Nx, Ny, state_grid_map, kk, constraints=None, trajectories=None,  optimal_policy=None):
    """
    Plots the skeleton of the grid world
    :param ax:
    :return:
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    act_dict = {}
    dx_dy = 0.1
    act_dict[0] = (0, -dx_dy)
    act_dict[1] = (0, dx_dy)
    act_dict[2] = (-dx_dy, 0)
    act_dict[3] = (dx_dy, 0)

    
    fig, ax = plt.subplots()
    
    for i in range(Ny + 1):
        ax.plot(np.arange(Nx + 1) - 0.5, np.ones(Nx + 1) * i - 0.5, color='k')
    for i in range(Nx + 1):
        ax.plot(np.ones(Ny + 1) * i - 0.5, np.arange(Ny + 1) - 0.5, color='k')
        # IPython.embed()

    y_grid = np.arange(Ny)
    x_grid = np.arange(Nx)
    # IPython.embed()
    cnt = 0
    for y in y_grid:
        for x in x_grid:
            if y==0 and x==0:
                cnt += 1
                continue
            else:
                act = optimal_policy[cnt]
                tup = act_dict[act]
                ax.arrow(x - tup[0], y - tup[1], tup[0], tup[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
                cnt += 1

    ax.plot(0,0,'x',color='white', markersize=15, markeredgewidth=5)




    data = np.ones((Ny, Nx)) 
    
        # data[cnstr[0], cnstr[1]] = 10

    # data = np.zeros((Ny, Nx)) 
    for i in trajectories:
        cnstr = state_grid_map[i[0]]
        data[Ny - 1 - cnstr[1], cnstr[0]] += 1
        # data[cnstr[0], cnstr[1]] = 10
    for ind in np.nonzero(constraints)[0]:
        cnstr = state_grid_map[ind]
        data[Ny - 1 - cnstr[1], cnstr[0]] = 0

    
    # ax.axes.xaxis.set_ticks([])
    # ax.axes.yaxis.set_ticks([])
    # cmap = colors.ListedColormap(['white', 'red'])
    # bounds = [0, 10, 20]
    # norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # im1 = plt.imshow(data2, cmap=plt.cm.viridis)
    cmap = cmap=plt.cm.viridis#mpl.cm.get_cmap("OrRd").copy()
    cmap.set_under('r')


    # cmap = colors.ListedColormap(['white', 'red'])
    bounds = [0, 10, 20]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    im = plt.imshow(data, vmin=0.1, cmap=cmap)


    # IPython.embed()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(False)
    plt.savefig('plots/grid_traj' + str(kk) + '.png')

    # plt.show()

    # return ax

def plot_posterior(Nx, Ny, posterior, ii, constraints, unc_flag=False, state_grid_map=None):
    """
    Plots the skeleton of the grid world
    :param ax:
    :return:
    """
    if not unc_flag:
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        cnt = 1
        for i in range(Ny):
            for j in range(Nx):
                ax = fig.add_subplot(Ny, Nx, cnt)
                if cnt-1 in constraints:
                    ax.hist(posterior[cnt-1], color = 'red')
                else:
                    ax.hist(posterior[cnt-1])
                plt.xlim([-0.5, 1.5])
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(4) 
                plt.yticks([], [])
                cnt += 1
    else:

        fig, ax = plt.subplots()
        
        for i in range(Ny + 1):
            ax.plot(np.arange(Nx + 1) - 0.5, np.ones(Nx + 1) * i - 0.5, color='k')
        for i in range(Nx + 1):
            ax.plot(np.ones(Ny + 1) * i - 0.5, np.arange(Ny + 1) - 0.5, color='k')


        ax.plot(0,0,'x',color='white', markersize=15, markeredgewidth=5)

        data = np.zeros((Ny, Nx)) 
       
        for ind in range(Ny * Nx):
           
            cnstr = state_grid_map[ind]
            # IPython.embed()
            # if np.sum(posterior[ind]) == 0:
            #     theta = 0
            # else:
            #     theta = np.sum(posterior[ind])/len(posterior[ind])
            # print(theta * (1-theta))
            theta = posterior[ind]
            data[Ny - 1 - cnstr[1], cnstr[0]] = theta * (1-theta)


        # IPython.embed()
        cmap=plt.cm.viridis # mpl.cm.get_cmap("OrRd").copy()
       
        im = plt.imshow(data, vmin=0.0, vmax=0.25, cmap=cmap)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        # IPython.embed()
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # plt.colorbar()
        ax.grid(False)

        plt.savefig('plots/posterior_var'+ str(ii) +'.png')
        # plt.show()



def plot_grid_mean_constr(Nx, Ny, state_grid_map, kk=0, constraints_plot=None, mean_constraints=None, trajectories=None,  optimal_policy=None):
    """
    Plots the skeleton of the grid world
    :param ax:
    :return:
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    act_dict = {}
    dx_dy = 0.1
    act_dict[0] = (0, -dx_dy)
    act_dict[1] = (0, dx_dy)
    act_dict[2] = (-dx_dy, 0)
    act_dict[3] = (dx_dy, 0)

    
    fig, ax = plt.subplots()
    
    for i in range(Ny + 1):
        ax.plot(np.arange(Nx + 1) - 0.5, np.ones(Nx + 1) * i - 0.5, color='k')
    for i in range(Nx + 1):
        ax.plot(np.ones(Ny + 1) * i - 0.5, np.arange(Ny + 1) - 0.5, color='k')
        # IPython.embed()

    # y_grid = np.arange(Ny)
    # x_grid = np.arange(Nx)
    # IPython.embed()
    # cnt = 0
    # for y in y_grid:
    #     for x in x_grid:
    #         if y==0 and x==0:
    #             cnt += 1
    #             continue
    #         else:
    #             if cnt in np.nonzero(constraints_plot)[0]:
    #                 cnt+=1
    #             else:
    #                 act = optimal_policy[cnt]
    #                 tup = act_dict[act]
    #                 ax.arrow(x - tup[0], y - tup[1], tup[0], tup[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
                    # cnt += 1

    ax.plot(0,0,'x',color='white', markersize=15, markeredgewidth=5)




    data = np.zeros((Ny, Nx)) 
    
        # data[cnstr[0], cnstr[1]] = 10

    # data = np.zeros((Ny, Nx)) 
    # for i in trajectories:
    #     cnstr = state_grid_map[i[0]]
    #     data[Ny - 1 - cnstr[1], cnstr[0]] += 1
       
    # IPython.embed()
    for ind in np.nonzero(mean_constraints)[0]:
        # IPython.embed()
        cnstr = state_grid_map[ind]
        data[Ny - 1 - cnstr[1], cnstr[0]] = mean_constraints[ind]

    
    # ax.axes.xaxis.set_ticks([])
    # ax.axes.yaxis.set_ticks([])
    # cmap = colors.ListedColormap(['white', 'red'])
    # bounds = [0, 10, 20]
    # norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # im1 = plt.imshow(data2, cmap=plt.cm.viridis)
    cmap = cmap=plt.cm.viridis # mpl.cm.get_cmap("OrRd").copy()
    # cmap.set_under('r')


    # cmap = colors.ListedColormap(['white', 'red'])
    # bounds = [0, 10, 20]
    # norm = colors.BoundaryNorm(bounds, cmap.N)
    im = plt.imshow(data, vmin=0.0, vmax=1, cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    # IPython.embed()
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # plt.colorbar()
    ax.grid(False)
    plt.savefig('plots/mean_results' + str(kk) + '.png')
    # plt.show()

