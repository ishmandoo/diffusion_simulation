from simulator import *
from scipy import stats
import pickle
import copy
from multiprocessing import Process
from argparse import ArgumentParser

def getDiffusivityQuadratic(x):
    return 0.5*(0.2 + 2. * x)**2 

def getDiffusivityLinear(x):
    return 1. + 1. * x

def getNormalizedLinearDiffusivityFunction(gamma):
    sqrt2 = np.sqrt(2)
    return lambda x: ((sqrt2)/(gamma+1)) + (sqrt2*(gamma-1)/(gamma+1)) * x

    

def getLinearDiffusivityFunction(m):
    return lambda x: .1+((m-.1)*x)

    
def getNormalizedQuadraticDiffusivityFunction(gamma):
    sqrtg = np.sqrt(gamma)
    return lambda x: 0.5*(((2)/(sqrtg+1)) + (2*(sqrtg-1)/(sqrtg+1)) * x)**2


def drawPaths(sim, n_steps):
    print(f"starting simulation {sim.name}")
    sim.setStartingXsDelta(0.5) 
    

    
    history = np.zeros((n_steps, sim.n))
    net_exits_history = np.zeros((n_steps, sim.n))
    total_exits_history = np.zeros((n_steps, sim.n))
    net_reflections_history = np.zeros((n_steps, sim.n))
    total_reflections_history = np.zeros((n_steps, sim.n))


    #plt.figure()

    for i in range(n_steps):

        sim.stepNum()
        history[i,:] = sim.xs
        net_exits_history[i,:] = (sim.pos_exits - sim.neg_exits)
        total_exits_history[i,:] = (sim.pos_exits + sim.neg_exits)
        net_reflections_history[i,:] = (sim.pos_reflections - sim.neg_reflections)
        total_reflections_history[i,:] = (sim.pos_reflections + sim.neg_reflections)
        #if i in frames:
        #    hist, pos = np.histogram(sim.xs, bins=25, density=False, range=(0.,100.))
        #    plt.plot(pos[:-1]+np.diff(pos)/2., hist, label=str(i))
    '''
    plt.legend()
    plt.xticks([0.,25.,50.,75.,100.])
    plt.yticks([])
    plt.title(sim.name)
    plt.xlabel("x")
    plt.ylabel(r"$\rho(x)$")
    '''
    
    '''
    plt.figure()
    plt.title(sim.name)
    plt.plot(history)
    plt.xlabel("t")
    plt.ylabel("x")

    plt.figure()
    plt.plot(net_exits_history)
    plt.title(sim.name)
    plt.xlabel("t (steps)")
    plt.ylabel(r"net exits")
    
    plt.figure()
    plt.plot(total_exits_history)
    plt.title(sim.name)
    plt.xlabel("t (steps)")
    plt.ylabel(r"total exits")
    
    plt.figure()
    plt.plot(total_reflections_history)
    plt.title(sim.name)
    plt.xlabel("t (steps)")
    plt.ylabel(r"total bounces")
    '''


    print(f'''Simulation {sim.name}:
    alpha {sim.alpha}
    mean {np.mean(sim.xs)}
    exit pos {sim.n_pos_exit}, neg {sim.n_neg_exit}
    net exit {sim.n_pos_exit - sim.n_neg_exit}
    center pos {sim.n_pos_center_cross}, neg {sim.n_neg_center_cross}
    center cross {sim.n_pos_center_cross - sim.n_neg_center_cross}
    interface fraction {(sim.n_pos_exit + sim.n_neg_exit + sim.n_pos_reflect + sim.n_neg_reflect)/float(sim.n * sim.n_steps)}
    exits {sim.n_pos_exit + sim.n_neg_exit}
    reflections {sim.n_pos_reflect + sim.n_neg_reflect}
    interface events {sim.n_pos_exit + sim.n_neg_exit + sim.n_pos_reflect + sim.n_neg_reflect}
    steps {sim.n_steps}
    ''')
    
    #sim.sanitize()
    #pickle.dump(sim, open(f"length_prob_{name}.p", "wb"))
    #plt.show()



parser = ArgumentParser()
parser.add_argument("-w", "--width", dest="w", type=int,
                help="The domain width")
parser.add_argument("-n", "--n_parts", dest="n", type=int,
                help="The number of particles")
parser.add_argument("-t", "--n_steps", dest="t", type=int,
                help="The number of steps in the simulation")
parser.add_argument("-p", "--prob", dest="p", type=float,
                help="The electrode absorption probability")

args = parser.parse_args()


sim = Simulator(name=f"width {args.w}, prob {args.p}", 
    n=4, 
    dt=1., 
    w=args.w, 
    alpha=1., 
    count_flux=True, 
    bc=BC.WRAP, 
    electrode_absorb_prob=args.p,
    getDiffusivity=getNormalizedLinearDiffusivityFunction(100)
    )

drawPaths(sim, args.t)
