
import os
import importlib.util

# Helper function to import the generated numerical python module
# from the generated script file
def import_source(source_dir):
    model_name = 'sym_model'
    source_file = os.path.join(source_dir, '%s.py'%model_name)
    # importing the topology source file 
    spec  = importlib.util.spec_from_file_location(model_name, source_file)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

dir_name = os.path.abspath(os.path.dirname(__file__))

for subdir in os.listdir(dir_name):
    file_ = os.path.join(dir_name, subdir)
    if os.path.isdir(file_) and not subdir.startswith('.'):
        script = os.path.join(file_, 'sym_model.py')
        print('Running file : %s'%script)
        try:
            import_source(file_)
        except FileNotFoundError:
            print('File Not Found!! \nPassing')
            pass
        print('\n')
