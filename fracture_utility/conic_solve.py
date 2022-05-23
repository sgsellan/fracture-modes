# Include existing libraries
import numpy as np
import sys

# Mosek for the conic solve
import mosek


def conic_solve(D, M, Us, c, d, verbose=False):
    # This uses Mosek to solve the conic problem
    #           argmin     ||Du||_{2,1}
    #           s.t.       u' M Us = 0
    #           and        u' M c = 1
    # 
    # Unfortunately MOSEK's conic API is quite complicated so we actually have 
    # to write it as
    #          argmin    sum (z_e)                <--- linear
    #           s.t.     ze >= sqrt(sum(Yd^2))    <--- cone (linear if d=1)
    #           and      Y = Du                   <--- linear
    #           and      u' M Us = 0              <--- linear
    #           and      u' M c = 1               <--- linear


    # From Mosek template, apparently we have to add this
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    # Mosek boilerplate
    with mosek.Env() as env:
        if verbose:
            env.set_Stream(mosek.streamtype.log, streamprinter)
        with env.Task(0,0) as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log, streamprinter)

            # Dimensions and degrees of freedom
            p = D.shape[0] // d
            n = D.shape[1]
            ndofs = n+ p*d+p
            task.appendvars(ndofs)
            task.putvarboundlistconst([*range(ndofs)], mosek.boundkey.fr, 0., 0.)
            task.appendcons(p*d+n+1+len(Us))

            # Objective function
            task.putclist([*range(   n+ p*d,    n+ p*d+p)],
                    [1.] * (p))

            

            #Set up equality constraint Y = Du
            nrows = 0
            #D
            task.putaijlist(nrows+D.row, D.col, D.data)
            #-Y
            task.putaijlist([*range(nrows, nrows+ p*d)],
                [*range(   n,    n+ p*d)], [-1.]*( p*d))
            task.putconboundlistconst([*range(nrows,nrows+ p*d)],
                mosek.boundkey.fx, 0., 0.)
            nrows +=  p*d



            # Set up orthogonality constraint wrt Us
            for U in Us:
                UtM = M*U
                task.putaijlist([nrows]*len(UtM), [*range(  0,    n)], UtM)
                task.putconbound(nrows, mosek.boundkey.fx, 0., 0.)
                nrows += 1
            # Set up norm-1 constraint wrt c
            ctM = M*c
            task.putaijlist([nrows]*len(ctM), [*range(  0,    n)], ctM)
            task.putconbound(nrows, mosek.boundkey.fx, 1., 1.)
            nrows += 1


            # Set up cone ze >= sqrt(Yes^2)
            for e in range(p):
                coneinds = [   n+ p*d+e]
                for dim in range(d):
                    coneinds.extend([   n+ p*dim+e])
                task.appendcone(mosek.conetype.quad, 0., coneinds)

            # Solve
            task.putobjsense(mosek.objsense.minimize)
            task.optimize()
            xx = [0.] * ndofs
            task.getxx(mosek.soltype.itr, xx)

            return np.asarray(xx)[0:n] # Extract just the u part from the solution