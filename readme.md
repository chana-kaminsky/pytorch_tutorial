from https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=1
by Python Engineer


for import torch to work:
- install anaconda and pytorch
- make a conda environment and install pytorch in it 
- make sure pytorch exstention is installed in vs code
- make sure file in vscode is in a folder with trusted 
    authors, so it's not running restricted mode
- go to view -> command palette -> python: select interpreter,
    and choose the conda environment with pytorch
- also might need to make sure that vscode is running the
    same version of python that the conda environment is

-- also... add conda to environment variables by adding these to path:
    C:\Users\Chana\anaconda3\
    C:\Users\Chana\anaconda3\Scripts\

-- AND open vscode from an anaconda shell with 'code .' command
-- in vscode, use powershell as cmd line, can run program fm it
