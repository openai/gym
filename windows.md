### Windows native
In Powershell
```
git clone gym
pip install -e .
choco install ffmpeg
choco install golang
pip install go-vncdriver
```

### Development

```
pip install nose2
pip install mock
```

* Download the windows version of swig with the pre-built executable from [here](http://www.swig.org/download.html) and extract it into your "Program Files (x86)" directory.

* Add swig to your path
```
setx PATH "$env:path;C:\Program Files (x86)\swigwin-3.0.10" -m
```

[Install MKL](https://software.intel.com/en-us/articles/free-mkl) from Intel.

Click the _Intel® Software Development Products Registration Center_ link in the email they send and download MKL from there. (613MB)

```
pip install -e .[all]
```
