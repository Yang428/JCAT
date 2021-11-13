
tracker_label = 'JCAT';

tracker_command = generate_python_command('vot_wrapper', ...
    {'/home/wcz/Yang/JCAT/pytracking/', ...
    '/media/wcz/datasets/yang/vot-toolkit/native/trax/support/python/',...
   '/home/wcz/Yang/JCAT/' });

tracker_interpreter = 'python';

tracker_linkpath = {'/media/wcz/datasets/yang/vot-toolkit/native/trax/build/'};
