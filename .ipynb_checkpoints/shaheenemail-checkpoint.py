import subprocess, sys, os

class Email:
    def __init__(self, receiver):
        self.receiver = receiver
    def send(self, subject, text='', file=None):
        if file:
            subprocess.Popen('echo "%s" | mail -s "%s" -a %s %s' % (text, subject, file, self.receiver), shell=True)
        else:
            subprocess.Popen('echo "%s" | mail -s "%s" %s' % (text, subject, self.receiver),shell=True)


if __name__=='__main__':
    mail = Email('imjoewang@gmail.com')
    subject = "Email test"
    text = "Hi,\n\nlet me test your email!\n\nBest regards\n\nShaheen"
    mail.send(subject,text)


