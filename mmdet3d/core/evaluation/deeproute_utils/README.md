This is an evaluation tool for detection, tracking and prediction.
Its instruction is shown as follows:

0. Install iou_cuda:

		(1) cd bev_ops
		(2) python setup.py install
		(3) pip install .



1. Detection:

		(1) evaluation:
		(a) cd deeproute
		(b) python compute_occl_ignore.py 15
		(c) python evaluation_detection.py 15 (or python evaluation_detection_v2.py 15)
	
		Note for evaluation: 
		1) step (b) can be ignored when you have generated occlusion distribution of gt and gt is not changed. 
		2) you can choose a more suitable number of processes in step (b) or (c) rather than a fixed number (e.g., 15)
		3) do not foget to modify the path in compute_occl.py and evaluation_detection.py.
		4) evaluation_detection_v2.py is much faster than evaluation_detection.py.

		(2) visualization:
		(a) cd deeproute && python backend_detection.py main
		(b) cd frontend_detection && python -m http.server
		(c) input url: http://127.0.0.1:8000
		(d) modify path in panel

		Note for visualization:
		1) focusRange: set the range you want to focus, (e.g., 40, 80, 120, full)
		2) type: which class you want to focus on (e.g., car, pedestrain, cyclist, total. default: car)
		3) PCformat: enable two format of pointCloud (default: nuympy)
		4) showError: show missing and fp via yellow and red boxes respectively
		5) autoplay: space key to control autoplay
		6) showDistance: draw circles according to distance



2. Tracking:

		Dependencies: mpi4py
		install from pip: pip install mpi4py
		or install from source: 
			(a) download package from https://pypi.python.org/pypi/mpi4py 
			(b) tar -xvzf mip4py-X.Y.Z.tar.gz
			(c) cd miprpy-X.Y.Z
			(d) python setup.py build
			(e) test from command: from mpi4py import MPI
	
		(1) evaluation:
		(a) cd deeproute
		(b) mpirun -np 3 python generate_velocity.py      (3 is faster than other numbers in my system)
		(c) python evaluation_tracking.py 15
	
		Note for evaluation:
		1) step (b) can be ignored when you have generated velocity and its confidences of gt and gt is not changed. 
		2) do not foget modify the path in evaluation_tracking.py
		3) you can choose a more suitable number of processes in step (c) rather than a fixed number (e.g., 15)

		(2) visualization:
		(a) cd deeproute && python backend_tracking.py main
		(b) cd frontend_tracking && python -m http.server
		(c) input url: http://127.0.0.1:8000
		(d) modify path in panel

		Note for visualization:
		1) focusRange: set the range you want to focus, (e.g., 40, 80, 120, full)
		2) type: which class you want to focus on (e.g., car, pedestrain, cyclist, total. default: car)
		3) PCformat: enable two format of pointCloud (default: nuympy)
		4) showIDS: only show those timeStamp that happens ID switch
		5) showMode: one color for one id when showMode is false
		6) autoplay: space key to control autoplay
		7) showVelocity: show Velocity when showVelocity is true
		8) showDistance: draw circles according to distance
		9) showVelocityBox: show Velocity with respect to box center when showVelocityBox is true, whereas show Velocity with respect to icp when showVelocityBox is false 



3. Prediction:

        (1) evaluation
        (a) modify path and has_priority parameter in configs/prediction.eval.path.config
        (b) python evaluation_prediction.py

        Note for evaluation:
        1) operate(2) will run prediction_data_manager to generate related attributes of groundtruth 
            if you are the first time to run this evaluation and this procedure will take about 1 hour.
            Hence, you can copy attributes from jiamiao If you use the same data(2019.4.12_rain_1)
        2) do not forget to delete corresponding pickle file when you change attribute threshold.
            Here provide a list: 
            turn_thresh -> gt_turn.pkl; 
            accelerate_thresh -> gt_acclerate.pkl
        
        (2) gt visualization   
        (a) modify path in configs/prediction.eval.path.config
        (b) set show_mode in gt_visualizer.py where 1 for cut-in and 2 for turn...
        (c) python gt_visualizer.py in prediction_visualization
        
        (3) a bird-eye-view visualization
        (a) modify path in configs/prediction.eval.path.config
        (b) set show_mode, show_eror, show_time, has_priority in visualizer.py where 0 for all, 1 for cut-in and 2 for turn...
        (c) python visualizer.py in prediction_visualization
    
        (4) prediction results visualzation
        (a) python prediction_backend.py main
        (b) cd frontend_prediction && python -m http.server
        (c) modify path in panel
        (d) modify behavior (all, cut_in, turn, on_lane, off_lane, accelerate, decelerate)


If any problem or suggestion, please contact Jiamiao Xu (jiamiaoxu@deeproute.ai)

Best，

Jiamiao





