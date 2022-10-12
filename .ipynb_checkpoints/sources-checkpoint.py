import os
from platform import platform
from shaheen import single_sim, sim_list

class source(object):

    def __init__(self, sep='+'):
        self.sep = sep

    def __str__(self):
        s = self.name
        for arg in self.args:
            s += self.sep
            s += arg
        return s

    
class cw(source):
    def __init__(self, wl, m=3, n=1e6):
        """
        :param wl: wavelength
        :param m: m-duration
        :param n: n-duration
        """
        source.__init__(self)
        self.name = 'cw'
        self.args = [str(wl),str(m), str(n)]

class spl(source):
    def __init__(self, wl, start=0):
        """
        :param wl: central frequency 1 / wl
        :param start: starting time
        """
        source.__init__(self)
        self.name = 'spl'
        self.args = [str(wl),str(start)]

class fcb(source):
    def __init__(self, wl0, wl1, N=5, n=3,m=100):
        """
        :param wl0:
        :param wl1:
        :param N:
        :param m:
        :param n:
        """
        source.__init__(self)
        self.name = 'fcb'
        self.args = [str(wl0),str(wl1), str(N),str(n), str(m)]

class run_fdtd():
    def __init__(self,
                 box_size,
                 number_of_cells,
                 jobid='meta',
                 exefile='nano',
                 dim = 2,
                 input_file='input.txt',
                 tfsf_box=None,
                 total_steps = 10000,
                 io_screen = 1000,
                 source = spl(0.5),
                 base_dir='res',
                 erg_io=10000,
                 upml=None,
                 media_filename='geometry.in',
                 platform='shaheen',
                 procs=[8,8,4],
                 shaheen_kargs={},
                 **kargs
                 ):
        """
        :param box_size: size of simulation box, array[dim]
        :param number_of_cells: number of cells on simulation grid, array[dim]
        :param exefile: default 'nano'
        :param dim: 2 or 3; default 2
        :param input_file: default 'input.in'
        :param tfsf_box: tfsf_box if needed; default None
        :param total_steps: default 10000
        :param io_screen: default 1000
        :param source: default spl; Should contain .__str__()
        :param base_dir: result directory
        :param erg_io: default 10000
        :param upml: set up upml layer; default None.
        :param media_filename: file to set up your geometry; default: 'geometry.in'
        :param kargs: additional key arguments to be delivered directly to exe string
        """
        self.pargs = []
        self.args = {}
        self.shaheen_kargs = shaheen_kargs
        self.exe = exefile
        self.dim = dim
        self.jobid=jobid
        self.makefile = 'makefile'
        self.args['box_size'] = ' '.join([str(x) for x in box_size])
        self.args['number_of_cells'] = ' '.join([str(x) for x in number_of_cells])
        self.args['procs']=' '.join([str(x) for x in procs])
        self.args['waveform'] = str(source)
        self.args['erg_io'] = str(erg_io)
        self.args['io_screen'] = str(io_screen)
        self.args['total_steps'] = str(total_steps)
        self.args['base_dir'] = base_dir
        self.args['media_filename'] = media_filename
        self.platform=platform
        # if tfsf were setted_up
        if tfsf_box is not None:
            self.args['tfsf_box'] = ' '.join([str(x) for x in tfsf_box])
            self.args['upml'] = ' '.join([str(x) for x in upml])
        #merge two dictionaries
        self.args = {**self.args, **kargs}
        #positional arguments
        self.pargs.append(self.get_workstation())
        self.pargs.append(input_file)


    def get_fdtd_str(self, sep=' ', end='\n'):
        s = sep.join(self.pargs)
        for key, value in self.args.items():
            s += sep
            s += key + sep + value
        logpath=os.path.join(self.args['base_dir'],'nanocpplog.txt')
        s += ' >> '
        s += logpath
        #s += end
        return s

    def get_workstation(self):
        if self.platform=='shaheen':
            self.workstation = 'shaheen'
            self.makefile = 'makefile_shaheen'
            return 'srun %s' % self.exe
        else:
            self.workstation = 'local'
            return 'mpiexec -n 24 ./%s' % self.exe

    def compile(self):
        self.get_workstation()
        os.system('make -f %s' % self.makefile)

    def run(self, output='', exe_string=None):
        if exe_string:
            s = exe_string
        else:
            s = self.get_fdtd_str()
        if output:
            s = s.rstrip()
            s += '> %s' % output
        if self.workstation == 'local':
            print(s)
            os.system(s)
        elif self.workstation == 'shaheen':
            sim = single_sim(self.jobid, [s], **self.shaheen_kargs)
            sims = sim_list([sim])
            sims.clean()
            sims.submit()
            sims.check_sims(5)



if __name__=='__main__':
    st = fcb(wl0=0.5,wl1=1.0)
    job = run_fdtd([20,30],[100,100])
    fdtd_string = job.get_fdtd_str()
    print(fdtd_string)
