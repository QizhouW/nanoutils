# ------------------------------------------------

# Maxim Makarenko
# Generalized Shaheen processing code
# Jan 2019

# ------------------------------------------------

# IMPORT DEPENDENCIES

import numpy as np
from matplotlib import pyplot as plt
import os
import time
import scipy.io as sio
import shutil as sh
import sys
import subprocess
from shaheenemail import Email


# Special shaheen exceptions
class ShaheenException(Exception):
    pass
# raise if job close to time limit
class TimeLimitWarning(ShaheenException):
    def __init__(self, job_id, suppl_info=''):
        self.job_id = job_id
        # supplememtary info string. Job name, time and etc.
        self.suppl_info = suppl_info

    def __str__(self):
        str = 'Job %s is close to shaheen time limit!\n' % self.job_id
        str += self.suppl_info
        return str

    def send_by_email(self, receiver):
        subject = 'Job Time Limit Warning'
        email = Email(receiver)
        subject = 'Job close to time limit!'
        str = 'Job %s has high risk of failure!\n' % self.job_id
        str += self.suppl_info
        email.send(subject, str)

# raise if job have been finished, but probably not started!
class JobFailureWarning(ShaheenException):
    def __init__(self, job_id, suppl_info=''):
        self.job_id = job_id
        # supplememtary info string. Job name, time and etc.
        self.suppl_info = suppl_info

    def __str__(self):
        str = 'Job %s has high risk of failure!\n' % self.job_id
        str += self.suppl_info
        return str

    def send_by_email(self, receiver):
        subject = 'Job Failure Warning'
        email = Email(receiver)
        subject = 'Job Failure Warning!'
        str = 'Job %s has high risk of failure!\n' % self.job_id
        str += self.suppl_info
        email.send(subject, str)


# Manipulate shaheen queue
class ShaheenQueue:
    def __init__(self, username):
        self.username = username

    def squeue(self):
        output_dict = {}
        pipe = subprocess.Popen(['squeue','-u',self.username], stdout=subprocess.PIPE)
        out, err = pipe.communicate()
        out = out.decode()
        for string in out.split('\n')[1:]:
            if not string:
                continue
            params = string.split()
            #ID
            output_dict[params[0]] = params[:]

        return output_dict

    def is_sim_in_queue(self, id):
        state = False
        jobs = self.squeue()
        if id in jobs:
            state = True
        return state

    def is_sim_running(self, id):
        state = False
        jobs = self.squeue()
        if id in jobs and jobs[id][4] == 'R':
            state = True
        return state

    def get_sim_parameter(self, id, pname):
        jobs = self.squeue()
        if id in jobs:
            return {
                'JOBID'      : jobs[id][0],
                'USER'       : jobs[id][1],
                'ACCOUNT'    : jobs[id][2],
                'NAME'       : jobs[id][3],
                'ST'         : jobs[id][4],
                'REASON'     : jobs[id][5],
                'START_TIME' : jobs[id][6],
                'TIME'       : jobs[id][7],
                'TIME_LEFT'  : jobs[id][8],
                'NODES'      : jobs[id][9]
            }[pname]
        else:
            return None

class ShaheenSim:
    def __init__(self, id, email = 'imjoewang@gmail.com'):
        self.username = self.get_username_from_home_dir()
        self.queue = ShaheenQueue(self.username)
        self.email = email
        if email is None:
            self.email = self.username
        self.sh_job_id = id
        self.sh_job_name = self.job_parameter('NAME')
        self.sh_job_status = self.check_job_status()

    def get_username_from_home_dir(self):
        return os.path.basename(os.environ['HOME'])

    def job_parameter(self,pname):
        return self.queue.get_sim_parameter(self.sh_job_id,pname)

    def check_job_status(self):
        #Missing
        job_status = 'M'
        if self.queue.is_sim_running(self.sh_job_id):
            #Running
            job_status = 'R'
        elif self.queue.is_sim_in_queue(self.sh_job_id):
            #Queue
            job_status = 'Q'
        return job_status

    def time_left(self):
        time_string = self.job_parameter('TIME_LEFT')
        if time_string and not time_string == 'INVALID':
            hms = time_string.split(':')
            seconds_remain = 0
            for i,t in enumerate(hms[::-1]):
                ##### hours, minutes, seconds
                seconds_remain += 60 ** i * int(t)
            return seconds_remain
        else :
            return None




class single_sim(ShaheenSim):

    def __init__(self, sim_name, sim_strings, queue='debug', outputdir='.shaheen', resdir='.', datadir='.', time='00:30:00', ntasks = 64, files_to_check=None, string_to_check=None, separator = " && ", print_warning=True, try_relaunch=True, **kargs):
        if not os.path.isdir('.shaheen'):
            os.mkdir('.shaheen')
        self.id = sim_name
        self.outputdir = os.path.abspath(outputdir)
        self.resdir = os.path.abspath(resdir)
        self.datadir = os.path.abspath(datadir)
        self.time = time
        self.ntasks = ntasks
        # check if simulation is over
        if files_to_check is None:
            files_to_check = []
        assert isinstance(files_to_check,list), 'Files to Check Must Be a List Input'
        self.files_to_check = [os.path.join(self.resdir,file) for file in files_to_check]
        self.string_to_check = string_to_check
        #################################################################################
        self.work_queue = queue  # queue of the simulation
        # All simulation strings (serial)
        self.str = sim_strings[0]
        for str in sim_strings[1:]:
            self.str += separator
            self.str += str  # string to execute simulation
        # internal data initialization
        self.job_name = ""
        self.job_name += "%s" % self.id  # add id of multiple dimensions
        self.outputfile = os.path.join(self.outputdir,self.job_name + '.out')
        self.errorfile = os.path.join(self.outputdir,self.job_name + '.err')
        self.testfile = self.outputfile
        self.testfileold = self.testfile[:-4] + '_old.out'
        if 'testfile' in kargs.keys():
            self.testfile = kargs['testfile']
            self.testfileold = self.testfile[:-4] + '_old.out'
        #what to check
        self.check = ['squeue']
        if self.string_to_check:
            self.check.append('string')
        if self.files_to_check:
            self.check.append('files')
        self.finished = False  # update simulation status
        self.started = False
        self.print_warning = print_warning
        self.try_relaunch = try_relaunch

    ############ execute batch file
    def submit_batch(self):
        # prepare batch file
        str1 = '#!/bin/bash\n'
        str1 = str1 + '#SBATCH--account=k1208\n'
        str1 = str1 + '#SBATCH--job-name=' + self.job_name + '\n'
        str1 = str1 + '#SBATCH--output=' + self.outputfile + '\n'
        str1 = str1 + '#SBATCH--error=' + self.errorfile + '\n'
        str1 = str1 + '#SBATCH--ntasks=%i\n' % self.ntasks
        str1 = str1 + '#SBATCH--time=%s\n' % self.time
        str1 = str1 + '#SBATCH--partition=' + self.work_queue + '\n'
        if self.work_queue == '72hours':
            str1 = str1 + '#SBATCH --qos=72hours\n'
        # add the execution command
        str = str1 + self.str + '\n'
        fname = os.path.join(self.outputdir, self.job_name + '.sh')
        f = open(fname, 'w')
        f.write(str)
        f.close()
        # execute it
        str = 'sbatch ' + fname
        pipe = subprocess.Popen(str, shell=True, stdout=subprocess.PIPE)
        out, err = pipe.communicate()
        out = out.decode()
        out.rstrip()
        id = out.split()[-1]
        ##### init shaheen simulation object
        ShaheenSim.__init__(self, id)


    # CHECK STATUS OF A SINGLE SIM
    # checks if self.testfile is created and then test for the occurrence of strg inside output file
    def check_sim(self):
        # check sim file is created
        status = {}
        for test in self.check:
            status[test] = self.check_for(test)

        if all(status.values()):
            self.finished = True

        time_left = self.time_left()
        ####### Raise some warnings in case job not in queue anymore bot other conditions is not satisfied
        if (status['squeue'] and not status.get('files',True)) or (status['squeue'] and not status.get('string',True)):
            raise JobFailureWarning(self.sh_job_id,"job name is %s, started = %s" % (self.sh_job_name, str(self.started)))
        if self.finished and not self.started:
            raise JobFailureWarning(self.sh_job_id,"job name is %s, started = %s" % (self.sh_job_name, str(self.started)))
        if time_left and time_left < 30:
            raise TimeLimitWarning(self.sh_job_id,"job name is %s" % self.sh_job_name)

    def check_for(self,test):
        if test == 'squeue':
            return not self.is_job_in_queue()
        elif test == 'files':
            return self.is_checkfiles_exist()
        elif test == 'string':
            return self.is_testfile_exist() and self.is_string_in_testfile_exist(self.string_to_check)

    def __str__(self):
        return self.job_name

    def is_testfile_exist(self):
        # if don't need to check for a test file
        if not self.string_to_check:
            return False
        if os.path.exists(self.testfile):
            return True
        else:
            return False

    def is_checkfiles_exist(self):
        # if don't need to check for a test file
        if not self.files_to_check:
            return False
        status = True
        for file in self.files_to_check:
            status = status and os.path.exists(file)
        return status

    def is_string_in_testfile_exist(self, strg_to_check):
        status = False
        if (os.path.isfile(self.testfile)):
            f = open(self.testfile, 'r')
            sfile = f.read()
            f.close()
            if strg_to_check in sfile:
                status = True
        return status

    def is_job_in_queue(self):
        status = False
        if not self.started:
            if self.check_job_status() == 'R':
                self.started =True
        if not self.check_job_status() == 'M':
            status = True
        return status

    def relaunch(self):
        self.finished = False  # update simulation status
        self.started = False
        self.print_warning = True
        self.submit_batch()



class sim_list(list):

    def __init__(self, pargs = []):
        """
        Basic class to handle simulation lists
        :param pargs: list of simulations
        """
        list.__init__(self,pargs)

    # checks all sim status every tt seconds and do not proceed until all sims are done
    # search occurrence of file named strg created in sim.testfile and in the res directory
    def check_sims(self,tt=5):
        """
        check if simulations are finished
        :param tt: time interval to check if simulation is finished
        :return: None
        """
        # check if sim finished
        loop=True
        limit=10000
        tm=0
        print('\n')
        while(loop):
            #check all sims against string strg in output file
            tmp=True
            for sim in self:
                try:
                    sim.check_sim()
                except (JobFailureWarning, TimeLimitWarning) as exc:
                    if sim.print_warning:
                        print(exc)
                        exc.send_by_email(sim.email)
                        ####### Print warning only one time!
                        sim.print_warning = False
                    if sim.try_relaunch and not sim.is_job_in_queue():
                        print('Trying to relaunch job %s' % sim.sh_job_name)
                        sim.relaunch()
                        print('Job %s was submitted' % sim.sh_job_id)
                        sim.try_relaunch = False
                tmp=tmp and sim.finished
            if tmp or tm>limit:
                loop=False
            else:
                time.sleep(tt)
                tm+=tt
        if tm>limit:
            print('no time')
            sys.exit(0)
            
    def submit(self):
        for sim in self:
            if sim.is_testfile_exist() or sim.is_checkfiles_exist():
                print('Testfile or one of the checkfiles does already exist, could be a problem with simulation checking! Use a clean function before!')
                exit(0)
        for sim in self:
            sim.submit_batch()
            # time.sleep(20)

    def __getitem__(self, item):
        retval = list.__getitem__(self,item)
        if isinstance(item, slice):
            retval = type(self)(retval)
        return retval

    def __str__(self):
        string = ''
        for sim in self:
            string += sim.__str__()
            string += '\n'
        return string
    # DESTRUCTOR
    def clean(self):
        #sim all finshed, rename files
        for sim in self:
            #rename output file
            if sim.is_testfile_exist():
                sh.move(sim.testfile,sim.testfileold)
                os.remove(sim.errorfile)
            #rename strg file
            if sim.files_to_check:
                if sim.is_checkfiles_exist():
                    for fname in sim.files_to_check:
                        name, ext = os.path.splitext(fname)
                        newfile = name + '_old' + ext
                        #print('cleaning '+fname+' '+newfile)
                        sh.move(fname,newfile)

if __name__ == '__main__':
    sims = sim_list()
    exe_strgs = []
    final_string = 'Finished!'
    for i in range(10):
        fname = 'test-%i.out' % i
        exe_strgs.append("echo %d" % i)
        exe_strgs.append('sleep %d' % int(3*i))
        exe_strgs.append('echo "%s"' % final_string)
        if i != 5:
            exe_strgs.append('touch %s' % fname)
        sim = single_sim('test_%d' % i,exe_strgs,files_to_check=[fname],string_to_check=final_string,try_relaunch=False)
        sims.append(sim)
    sims.clean()
    sims.submit()
    sims.check_sims(2)
    print(sims[:3])






