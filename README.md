# Analyzer
Akademicki projekt analizatora EKG realizowanego na przedmiocie "Dedykowane algorytmy diagnostyki medycznej"
[<img width="1231" alt="image" src="https://user-images.githubusercontent.com/22752828/202790800-678b2c61-5350-4608-bb71-5b9b9a2401cf.png">](https://coggle.it/diagram/Y3fhb71Eo51jevuX/t/-/6821abec65cb3b7c8fe64d795b9845c34a429eb27f34f7e216be326823b10820)
<img width="348" alt="image" src="https://user-images.githubusercontent.com/22752828/210072479-8b1378c2-1ad6-41ba-85e0-6bc8e585f10c.png">
<!-- https://www.researchgate.net/publication/332862150_A_statistical_designing_approach_to_MATLAB_based_functions_for_the_ECG_signal_preprocessing -->
## Onboarding
<!-- [How to install and use pip on macOS ](https://gist.github.com/haircut/14705555d58432a5f01f9188006a04ed) -->
[Automatic installation - mamba](https://mamba.readthedocs.io/en/latest/installation.html)

```sh
micromamba create -n "test" python=3.10 -c conda-forge
micromamba activate test
```
<!-- How to install on Debian Linux
```sh
sudo apt update
sudo apt install libffi-dev libsqlite3-dev zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev build-essential libreadline-dev wget libbz2-dev
wget https://www.python.org/ftp/python/3.9.16/Python-3.9.16.tgz 
tar -xf Python-3.9.16.tgz
cd Python-3.9.16
./configure --enable-optimizations
make -j 8
sudo make altinstall
``` -->
```python
pip3 install django  
pip3 install scipy
pip3 install django-crispy-forms
pip3 install wfdb
pip3 install padasip
mkdir static
python3 ./manage.py makemigrations
python3 ./manage.py migrate
python3 manage.py runserver
```

## Contributing

1. Clone the project
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new pull request
6. When feature is ready add reviewers and wait for feedback (at least one
   approve should be given and all review comments should be resolved)
