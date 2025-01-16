if continue_train:
    weights = os.listdir(self.results_folder)
    weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
    weight_path = max(weights_paths, key=os.path.getctime)

    print(f"Loaded weight: {weight_path}")

    milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")

    self.load(milestone)
    self.ema.ema_model.train()

print('begin to train!')
init_step = self.step 
epoch = init_step % self.epoch_iter
start_time = time.perf_counter()