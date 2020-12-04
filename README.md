# tomoPost
Postprocessing tools for tomography data


# Requirements

* python3 (recommendation 3.8)
* numpy
* scipy
* scikit-image
* pymesh


# Recommended instalation in conda environment

* Download and install Anaconda3. Example:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-MacOSX-x86_64.sh
chmod -x Anaconda3-2020.11-MacOSX-x86_64.sh
./Anaconda3-2020.11-MacOSX-x86_64.sh
... follow instructions ...
```

* Set up a conda environment. Example:
```bash
conda create -n tomo python=3.8
```

* Install required packages to created conda environment:
```bash
conda install -n tomo numpy
conda install -n tomo scipy scikit-image
```

* Activate conda created conda environment and install pymesh using pip: 
```bash
conda activate tomo	
pip install pymesh
```

* clone `tomoPost` to local directory
```bash
git clone https://github.com/vitst/tomoPost
```

Optional

	* Create executable and add it to env path in `.bashrc`
	```bash
	cd /local_directory/tomoPost
	mkdir bin
    cd bin
	```
	* Create a file `runTomo` with:
    ```bash
    #!/bin/bash
    python3 /path_to/local_directory/tomoPost/runPostTomo.py "$@"
    ```
	* Add it to env path in `.bashrc`
	```bash
	export PATH=/path_to/local_directory/tomoPost/bin:$PATH
	```

* test tomo

# Using WEKA segmentation

* Requirements
	* Fiji
	* beanshell

* Instructions that should work
	* install [Fiji](https://imagej.net/Fiji)
	* get [beanshell](https://github.com/beanshell/beanshell)
	* add path to Fiji and beanshell to env variable `CLASSPATH` (optionaly add it to `.bashrc`):
	```bash
	export CLASSPATH="$CLASSPATH:/path_to/bsh-2.0b4.jar:/path_to/Fiji.app/jars/*:/path_to/Fiji.app/plugins/*"
	```


