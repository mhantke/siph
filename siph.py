import numpy as np
from scipy.special import erf,erfinv
import scipy.optimize
import scipy.stats

# I(x): Integrated intensity of pixel as a function of the x-coordinate of the incident photon position (y=0)
I_x = lambda x, sigma: (erf((0.5-x)/(np.sqrt(2)*sigma)) + erf((0.5+x)/(np.sqrt(2)*sigma))) / 2.
# I_approx(x): This approximation is acceptable for sigma < 0.3
I_x_approx = lambda x, sigma: (1+erf((-abs(x)+0.5)/(np.sqrt(2)*sigma))) / 2.

# x(I): Inverse of the function I_approx(x) 
# (Certainly, it would be better if we found an expression for the inverse function of I(x) and used it instead)
x_I = lambda I, sigma: 0.5 - np.sqrt(2*sigma**2) * erfinv(2*I-1)
# d/dI{x(I)}: First derivative of x(I) 
diff_x_I = lambda I, sigma: -np.sqrt(2*sigma**2*np.pi) * np.exp(erfinv(2*I-1)**2)
# f(I) with y=0: Probability denstiy function of I for all -L/2 < x < L/2 and y=0
f_I_y0 = lambda I, sigma, L: abs(diff_x_I(I, sigma)) / L
# f(I): Probability denstiy function of I for all -L/2 < x < L/2 and all -L/2 < y < L/2
f_I = lambda I, sigma, L: 8*abs(x_I(I, sigma)) * f_I_y0(I, sigma, L) / L

# Create a histogram of pixel intensities I for testing purposes
def generate_test_I(sigma=0.1, L=3., nphotons=1000, noise_sigma=None): 
    x_sam = L*np.random.rand(nphotons) - L/2.
    y_sam = L*np.random.rand(nphotons) - L/2.
    d_sam = abs(np.array([x_sam, y_sam])).max(axis=0)
    values = np.zeros_like(d_sam)
    for ii,d_sami in enumerate(d_sam):
        values[ii] = I_x_approx(d_sami, sigma)
        if noise_sigma is not None:
            v = np.random.normal(values[ii], noise_sigma)
            while abs(v) > L/2.:
                xi_sam = L*np.random.rand() - L/2.
                yi_sam = L*np.random.rand() - L/2.
                di_sam = abs(np.array([xi_sam, yi_sam])).max()
                v = np.random.normal(di_sam, noise_sigma)
            values[ii] = v
    return values

I_arr = lambda nbins: np.linspace(-0.5, 1.5, nbins)
dI_arr = lambda nbins: 2.0/(nbins-1)

def histogram(values, nbins): 
    dI = dI_arr(nbins)
    H_test, edges = np.histogram(values, nbins, range=(-0.5-dI/2.,1.5+dI/2.))
    I_test = I_arr(nbins)
    return I_test, H_test

def generate_test_hist_I(sigma=0.1, L=3., nphotons=1000, noise_sigma=None, nbins=101):
    v = generate_test_I(sigma=sigma, L=L, nphotons=nphotons, noise_sigma=noise_sigma)
    I_test, H_test = histogram(v, nbins)
    return I_test, H_test


#_hist_I = lambda sigma, L, nphotons, nbins: f_I(I_arr(nbins), sigma, L) * dI_arr(nbins) * nphotons
_hist_I = lambda sigma, L, nphotons, nbins: f_I(I_arr(nbins), sigma, L) * (2./nbins) * nphotons
def hist_I(sigma, L, nphotons, noise_sigma=None, nbins=101):
    I = I_arr(nbins)
    dI = dI_arr(nbins)
    H = _hist_I(sigma, L, nphotons, nbins)
    # Fix lowest and highest intensity bin (avoid imprecision and trouble with inf)
    i0 = np.where( I+dI/2. >= I_x_approx(L/2., sigma) )[0][0]
    H[:i0] = 0
    H[i0] = (L**2 - (x_I(I[i0]+dI/2., sigma)*2)**2) / L**2 * nphotons
    # High intensities
    i1 =  np.where( (I+dI/2.) >= I_x_approx(0., sigma) )[0][0]
    H[i1] = (x_I(I[i1]-dI/2., sigma) * 2)**2 / L**2 * nphotons
    # Set remaining infinite values to zero
    H[np.isnan(H) | np.isinf(H)] = 0    
    if noise_sigma is not None:
        gauss = lambda i, s: np.exp(-i**2/(2.*s**2))
        G = gauss(I-0.5, noise_sigma)
        H = np.convolve(H, G/G.sum(), 'same')
    return I, H


def fit_hist(H, sigma=0.2, L=3, noise_sigma=0.1):
    nbins = len(H)
    nphotons = H.sum()
    x0 = [sigma, L, noise_sigma]
    func = lambda x: (hist_I(sigma=x[0], L=x[1], nphotons=nphotons, noise_sigma=x[2], nbins=nbins)[1] - H)/(np.finfo("float64").resolution + hist_I(sigma=x[0], L=x[1], nphotons=nphotons, noise_sigma=x[2], nbins=nbins)[1] + H)
    assert np.isnan(func(x0).any()) == False
    x_fit = scipy.optimize.leastsq(func, x0,
                                   args=(), Dfun=None,
                                   full_output=0, col_deriv=0,
                                   ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, 
                                   maxfev=0, epsfcn=None, factor=100, diag=None)[0]
    [sigma_fit, L_fit, noise_sigma_fit] = x_fit
    return sigma_fit, L_fit, nphotons, noise_sigma_fit

def fit_hist2(H, sigma=0.2, L=3, noise_sigma=0.1):
    nbins = len(H)
    nphotons = H.sum()

    # 1) Simple Gaussian fit for getting good estimate for noise_sigma
    I = I_arr(nbins)
    func = lambda noise_sigma: 1-scipy.stats.pearsonr(np.exp(-I[:nbins/4]**2/(2.*noise_sigma**2)), H[:nbins/4])[0]
    noise_sigma_fit = scipy.optimize.leastsq(func, noise_sigma)[0][0]
    # 2) Now fit of the other variables
    x0 = [sigma, L]
    S = I>=0.25
    func = lambda x: abs(hist_I(sigma=x[0], L=x[1], nphotons=nphotons, noise_sigma=noise_sigma_fit, nbins=nbins)[1] - H)[S]#/(np.finfo("float64").resolution + hist_I(sigma=x[0], L=x[1], nphotons=nphotons, noise_sigma=noise_sigma_fit, nbins=nbins)[1] + H)
    assert np.isnan(func(x0).any()) == False
    x_fit = scipy.optimize.leastsq(func, x0,
                                   args=(), Dfun=None,
                                   full_output=0, col_deriv=0,
                                   ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, 
                                   maxfev=0, epsfcn=None, factor=100, diag=None)[0]
    [sigma_fit, L_fit] = x_fit
    return sigma_fit, L_fit, nphotons, noise_sigma_fit


def test_fit_hist(sigma_lims, nphotons_lims, L_lims, noise_sigma_lims, nbins_lims, N_test_grid=2, do_plot=False):

    sigmas = np.linspace(sigma_lims[0], sigma_lims[1], N_test_grid)
    nphotonss = np.asarray(np.linspace(nphotons_lims[0], nphotons_lims[1], N_test_grid), dtype='i')
    Ls = np.linspace(L_lims[0], L_lims[1], N_test_grid)
    noise_sigmas = np.linspace(noise_sigma_lims[0], noise_sigma_lims[1], N_test_grid)

    nbinss = np.asarray(2 * np.round(np.linspace(nbins_lims[0], nbins_lims[1], N_test_grid)/2.) + 1, dtype='i')
    
    for nbins in nbinss:
        for sigma in sigmas:
            for nphotons in nphotonss:
                for L in Ls:
                    for noise_sigma in noise_sigmas:
                        
                        I_test, H_test = generate_test_hist_I(sigma=sigma, L=L, nphotons=nphotons, noise_sigma=noise_sigma, nbins=nbins)
                        sigma0 = np.random.rand()*sigma
                        L0 = np.random.rand()*L
                        nphotons0 = np.random.rand()*nphotons
                        noise_sigma0 = np.random.rand()*noise_sigma
                        sigma_fit, L_fit, nphotons_fit, noise_sigma_fit = fit_hist2(H_test, sigma=sigma0, L=L0, 
                                                                                    noise_sigma=noise_sigma0)
                        I_fit, H_fit = hist_I(sigma_fit, L_fit, nphotons_fit, noise_sigma_fit, nbins)
                        I_ideal, H_ideal = hist_I(sigma, L, nphotons, noise_sigma, nbins)

                        if do_plot:
                            import matplotlib
                            from matplotlib import pyplot as pypl
                            pypl.figure()
                            pypl.plot(I_test, H_test)
                            pypl.plot(I_fit, H_fit)
                            pypl.plot(I_ideal, H_ideal)
                            pypl.legend(['data', 'fit', 'ideal'])
                            pypl.ylim(0, H_ideal.max())
                            pypl.show()
                        
                        print "nbins, sigma, nphotons, L, noise_sigma"
                        print nbins, sigma, nphotons, L, noise_sigma
                        print nbins, sigma0, nphotons0, L0, noise_sigma0
                        print nbins, sigma_fit, nphotons_fit, L_fit, noise_sigma_fit
                        #err_max = 1.
                        #assert abs((sigma_fit-sigma)/sigma) < err_max
                        #assert abs((L_fit-L)/L) < err_max
                        #assert (nphotons_fit-nphotons)/nphotons < err_max
                        #assert abs((noise_sigma_fit-noise_sigma)/noise_sigma) < err_max
                        
                            
                            
