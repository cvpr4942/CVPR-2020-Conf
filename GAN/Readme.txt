Please follow the following step to run the code.

1- 	go to this website: https://github.com/bluer555/CR-GAN
	download the multi_PIE_crop_128 dataset
	pu the extracted dataset to this folde: /data/multi_PIE_crop_128/

2-	open selection.py
	set parameter selection using the desired method, e.g., selection = 'SP' 
	running selection.py results in a .txt file in /data/multi_PIE_crop_128/

3-      train using the selected samples in .txt file:
	open train.py
	set the .txt filename: default="data/multi_PIE_crop_128/selected_SP.txt") 
	run train.py results in two networks in '/pretrained_model/' folder (it takes few hours)

4-      open evaluate.py
	for evaluation on the test set first make sure that the trained networks are fed:
	load_pretrained_model(G_xvz, args.modelf, 'netG_xvz.pth')
	load_pretrained_model(G_vzx, args.modelf, 'netG_vzx.pth') 
	run evaluate.py
	 