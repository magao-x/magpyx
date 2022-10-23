from functools import partial
import multiprocessing as mp
from time import sleep

import numpy as np
try:
    import cupy as cp
except ImportError:
    logger.warning('Could not import cupy. You may lose functionality.')
    cp = None

from scipy.optimize import minimize
from scipy.ndimage import binary_erosion

from ..imutils import gauss_convolve, fft2_shiftnorm, ifft2_shiftnorm, center_of_mass, shift

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fdpr2')

def get_array_module(arr):
    if cp is not None:
        return cp.get_array_module(arr)
    else:
        return np

def forward_model(pupil, Eprobes, Eab):
    xp = get_array_module(Eab)
    Epupils = pupil * Eab * Eprobes # (k,y,x)
    Epupils /= np.mean(np.abs(Epupils),axis=(-2,-1))[:,None,None]
    Efocals = fft2_shiftnorm(Epupils, axes=(-2,-1)) # k simultaneous FFTs
    Ifocals = xp.abs(Efocals)**2
    return Ifocals, Efocals, Epupils
    
def get_err(Imeas, Imodel, weights):
    xp = get_array_module(Imeas)
    K = len(weights)
    t1 = xp.sum(weights * Imodel * Imeas, axis=(-2,-1))**2
    t2 = xp.sum(weights * Imeas**2, axis=(-2,-1))
    t3 = xp.sum(weights * Imodel**2, axis=(-2,-1))
    return 1 - 1/K * np.sum(t1/(t2*t3), axis=0)

def get_Ibar_model(Imeas, Imodel, weights):
    xp = get_array_module(Imeas)
    K = len(weights)
    t1 = xp.sum(Imeas*Imodel*weights, axis=(-2,-1))[:,None,None]
    t2 = xp.sum(weights*Imeas**2, axis=(-2,-1))[:,None,None]
    t3 = xp.sum(weights*Imodel**2, axis=(-2,-1))[:,None,None]
    return 2/K * weights * t1 / (t2 * t3**2) * (Imodel * t1 - Imeas * t3)

def get_grad(Imeas, Imodel, Efocals, Eprobes, Eab, A, phi, weights, pupil):
    xp = get_array_module(Imeas)
    
    # common gradient terms
    Ibar = get_Ibar_model(Imeas, Imodel, weights)
    Ehatbar = 2 * Efocals * Ibar
    Ebar = ifft2_shiftnorm(Ehatbar, axes=(-2,-1))
    
    # --- get Eab ---
    Eabbar = Ebar * Eprobes.conj()
    # get amplitude
    expiphi = np.exp(1j*phi)
    Abar = Eabbar * expiphi.conj()
    # get phase
    expiphibar = Eabbar * A
    phibar = xp.imag(expiphibar * expiphi.conj())

    # --- get E probe ---
    #(save for later for now)
    #Epbar = Ebar * Eab.conj()
    #phipbar = xp.imag(Epbar * Eprobes.conj())
    #abar = xp.sum(phipbar * phiprobes, axis=(-2,-1))
    
    # sum terms (better be purely real, should double check this!!!)
    gradA = xp.sum(Abar, axis=0).real
    gradphi = xp.sum(phibar, axis=0).real
    #grada = xp.sum(abar, axis=0).real
    
    '''if zmodes is not None: # project onto zernikes
        coeffsA = xp.sum(gradA*zmodes, axis=(-2,-1)) / zmodes[0].sum()
        coeffsphi = xp.sum(gradphi*zmodes, axis=(-2,-1)) / zmodes[0].sum()
        gradA = xp.sum(coeffsA[:,None,None]*zmodes,axis=0)
        gradphi = xp.sum(coeffsphi[:,None,None]*zmodes,axis=0)'''
    
    return gradA, gradphi

def get_sqerr_grad(params, pupil, mask, Eprobes, weights, Imeas, N, lambdap):
    
    xp = get_array_module(Eprobes)
    
    # CPU to GPU if needed
    if xp is cp and isinstance(params, np.ndarray):
        params = cp.array(params)
    
    # params to wavefront
    #param_a = params[0]
    params_amp = params[:N]
    params_phase = params[N:]
        
    #Eab = xp.zeros(mask.shape, dtype=complex)
    A = xp.zeros(mask.shape)
    phi = xp.zeros(mask.shape)
    A[mask] = params_amp
    phi[mask] = params_phase
    
    # probe
    #Eprobes = np.exp(1j*param_a*phiprobes)
    
    Eab = A * np.exp(1j*phi)
    #Eab[mask] = params_re + 1j*params_im
    
    #print(A.max(), phi.max())
    
    # forward model
    Imodel, Efocals, Epupils = forward_model(pupil, Eprobes, Eab)
    #return Imodel
    
    # lsq error
    #err = np.sum(weights * np.sqrt( (Imodel - Imeas)**2 ))
    err = get_err(Imeas, Imodel, weights) + lambdap * xp.sum(params**2) 
    
    # update mindict
    '''smoothing = None
    if mindict['iter'] > mindict['niter_zmodes']:
        zmodes = None
        smoothing = mindict['smoothing']
        if smoothing is None:
            pass
        else:
            smoothing = smoothing / mindict['iter_smoothing']**mindict['smoothing_exp']
            if smoothing < mindict['smoothing_min']:
                smoothing = mindict['smoothing_min']
            mindict['iter_smoothing'] += 1
    mindict['iter'] += 1'''
    
    # gradient
    # grada,
    gradA, gradphi = get_grad(Imeas, Imodel, Efocals, Eprobes, Eab, A, phi, weights, pupil)#[mask]
    # split into real, imaginary components
    grad_Aphi = xp.concatenate([#cp.asarray([grada,]),
                                gradA[mask],gradphi[mask]], axis=0) + 2 * lambdap * params
    
    # back to CPU
    if xp is cp:
        err = cp.asnumpy(err)
        grad_Aphi = cp.asnumpy(grad_Aphi)
    
    return err, grad_Aphi

def get_han2d_sq(N, fraction=1./np.sqrt(2), normalize=False):
    '''
    Radial Hanning window scaled to a fraction 
    of the array size.
    
    Fraction = 1. for circumscribed circle
    Fraction = 1/sqrt(2) for inscribed circle (default)
    '''
    #return np.sqrt(np.outer(np.hanning(shape[0]), np.hanning(shape[0])))

    # get radial distance from center

    # scale radial distances
    x = np.linspace(-N/2., N/2., num=N)
    rmax = N * fraction
    scaled = (1 - x / rmax) * np.pi/2.
    window = np.sin(scaled)**2
    window[np.abs(x) > rmax] = 0
    return np.outer(window, window)

def run_phase_retrieval(Imeas, fitmask, tol, reg, wreg, Eprobes, init_params=None, bounds=True):

    # centroiding here? probably not.

    N = np.count_nonzero(fitmask)

    # initialize pixel values if not given
    if init_params is None:
        fitsmooth = gauss_convolve(binary_erosion(fitmask, iterations=3), 3)
        init_params = np.concatenate([fitsmooth[fitmask],
                                      fitsmooth[fitmask]*0], axis=0)
    # compute weights?
    weights = 1/(Imeas + wreg) * get_han2d_sq(Imeas[0].shape[0], fraction=0.7)
    weights /= np.max(weights,axis=(-2,-1))[:,None,None]

    if bounds:
        bounds = [(0,None),]*N + [(None,None),]*N
    else:
        bounds = None

    # get probes

    # convert all to cupy arrays?
    Eprobes = cp.array(Eprobes, dtype=cp.complex128)
    Imeas = cp.array(Imeas, dtype=cp.float64)
    weights = cp.array(weights, dtype=cp.float64)
    fitmask_cp = cp.array(fitmask)

    errfunc = get_sqerr_grad
    fitdict = minimize(errfunc, init_params, args=(fitmask_cp, fitmask_cp,
                        Eprobes, weights, Imeas, N, reg),
                        method='L-BFGS-B', jac=True, bounds=bounds,
                        tol=tol, options={'ftol' : tol, 'gtol' : tol, 'maxls' : 100})

    # construct amplitude and phase
    phase_est = np.zeros(fitmask.shape)
    amp_est = np.zeros(fitmask.shape)

    phase_est[fitmask] = fitdict['x'][N:]
    amp_est[fitmask] = fitdict['x'][:N]

    return {
        'phase_est' : phase_est,
        'amp_est' : amp_est,
        'obj_val' : fitdict['fun'],
        'fit_params' : fitdict['x']
    }

# ------- multiprocessing -------

def _process_phase_retrieval_mpfriendly(fitmask, tol, reg, wreg, Eprobes, init_params, bounds, Imeas):
    return run_phase_retrieval(Imeas, fitmask, tol, reg, wreg, Eprobes, init_params=init_params, bounds=bounds)

def multiprocess_phase_retrieval(allpsfs, fitmask, tol, reg, wreg, Eprobes, init_params=None, bounds=True, processes=2, gpus=None):
    

    from functools import partial
    import multiprocessing as mp

    try:
        mp.set_start_method('spawn')
    except RuntimeError as e:
        logger.warning(e)

    # TO DO: figure out if this still needs to be sequential or can be the original multiprocess_phase_retrieval
    mpfunc = partial(_process_phase_retrieval_mpfriendly, fitmask, tol, reg, wreg, Eprobes, init_params, bounds)

    # There's a weird possibly memory-related issue that prevents us from simply
    # splitting the full allpsfs hypercube across the processes for multiprocessing
    # using a Pool object (small inputs work, large inputs result in GPU blocking).
    # So I've implemented my own queue and worker system here.

    # available GPUs for processing
    if gpus is None:
        gpus = [0,]

    # with a single GPU, parsing the config file will yield an int instead
    if isinstance(gpus, int):
        gpus = [gpus,]

    # assign multiple processes per GPU
    gpu_list = gpus * processes
    ntot = len(allpsfs)

    # make the queue and pass all jobs in
    mpqueue = GPUQueue(gpu_list, mpfunc)
    for (i, psfcube) in enumerate(allpsfs):
        #print(psfcube.shape)
        #print(i)
        mpqueue.add([i, psfcube])

    # check for completion every second
    while len(mpqueue.raw_results) < ntot:
        sleep(1)

    # get the results back in order
    allresults = mpqueue.get_sorted_results()
    mpqueue.terminate()

    #return allresults
    return allresults

class GPUWorker(mp.Process):
    def __init__(self, queue_in, queue_out, gpu_id, func):
        mp.Process.__init__(self, args=(queue_in,))
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.gpu_id = gpu_id
        self.func = func
        self.logger = mp.get_logger()
        
    def run(self):
        with cp.cuda.device.Device(self.gpu_id):
            while True:
                task_id, task = self.queue_in.get()
                
                logger.info(f'Worker {self} starting task {task_id}.')
                result = self.func(task)
        
                #self.logger.info(f'{self.gpu_id} got a task.')
                self.queue_out.put([task_id, result])
            
class GPUQueue(object):
    def __init__(self, gpu_list, func):
        
        self._queue_in = mp.Queue()
        self._queue_out = mp.Queue()
        self._results = []
        
        # make and start a worker for each entry in gpu_list
        self.workers = []
        for gpu in gpu_list:
            self.workers.append(GPUWorker(self._queue_in, self._queue_out, gpu, func))
        for w in self.workers:
            w.start()

    def add(self, task):
        self._queue_in.put(task)
            
    @property
    def raw_results(self):
        '''
        Grab the unsorted results from the worker queue
        '''
        while not self._queue_out.empty():
            self._results.append(self._queue_out.get_nowait())
        return self._results
    
    def get_sorted_results(self):
        results = self.raw_results
        sort_idx = np.argsort(np.asarray([r[0] for r in results]))
        return np.asarray([r[1] for r in results])[sort_idx]
            
    def terminate(self, timeout=5):
        for w in self.workers:
            w.terminate()
            w.join(timeout=timeout)
            w.close()
