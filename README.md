# SimInvVelRelocate

Developing tools to perform Simultaneous Inversion for surface wave phase Velocity and earthquake centroid parameters


## Installation
First clone this repository or download code on machine where you would like to setup.       

#### - Clone the repo
  
```
$ git clone https://github.com/SeisPider/SimInvVelRelocate.git
$ cd SimInvVelRelocate
```
 
#### - Install Python

[Windows](http://timmyreilly.azurewebsites.net/python-flask-windows-development-environment-setup/),[Mac](http://docs.python-guide.org/en/latest/starting/install/osx/),[Linux](https://docs.aws.amazon.com/cli/latest/userguide/awscli-install-linux-python.html)

### - Instal CPS330
[CPS330](http://www.eas.slu.edu/eqc/eqc_cps/CPS/CPS330.html)

#### - Install requirements.txt 
 
```
$ pip install -r requirements.txt
```

Above command will install all the dependencies of project.

## Folder structure

```shell
.
├── Measurement.Rayleigh.demo.py
├── data
│   └── 20190907224214
│       └── CB.WHN.00.BHZ.SAC
├── info
│   ├── refmodel
│   │   ├── 104_29
│   │   │   └── 114_30_cps.mod
│   │   └── 105_29
│   │       └── 114_30_cps.mod
│   └── test.info
├── measured
│   └── 20190907224214
│       └── Rayleigh
│           ├── Fiteness.Z.CB.WHN.00.message.info
│           ├── Fitness.Z.CB.WHN.00.pdf
│           ├── Fitted.Z.CB.WHN.00.waveform.SAC
│           ├── Fitted.param.Z.CB.WHN.00.info
│           ├── Ref.disp.CB.WHN.00.Rayleigh.info
│           └── Syn.CB.WHN.00.Z.SAC
├── requirements.txt
└── src
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-36.pyc
    │   ├── __init__.cpython-37.pyc
    │   ├── measurement.cpython-36.pyc
    │   ├── measurement.cpython-37.pyc
    │   ├── utils.cpython-36.pyc
    │   └── utils.cpython-37.pyc
    ├── measurement.py
    └── utils.py
```


## Data preprocessing


**Earthquake waveform:** decimate to 1Hz, remove instrumental response to velocity, badnpass filter.

> :warning: **Don't change the waveform phase:** Bandpass filter first and then decimate the data.


## Support

If you face any problem or issue in configuration or usage of SimInvVelRelocate  project as per the instruction documented above, Please feel free to communicate with SimInvVelRelocate Development Team


## Reference

**Xiao X**, Sun L., Wang X. and Wen, L. Simultaneous inversion for surface wave phase velocity and earthquake centroid parameters: methodology and application [J]. Journal of Geophysical Research: Solid Earth [Submitted]



