# sogang-behavior_finance-volatility
* Data
  * KOSPI index / net buying of private, institute, foreigner
  * get by scraping
* Code
  * main.py - 3 models
    * Linear
    * MC dropout
    * classification (rise or fall)
  * utils.py
    * all functions used in main.py is in utils.py
  * denoising.py
    * using CNN for denoise data
  * config.yml, search_space.json
    * for using nni
* Execution
  1. make denoising model using denoising.py and data
  2. select model, denoising or not, epochs
  3. modify search_space.json to fit the model (for example, dropout_rate is only used in MC dropout)
  4. run main via nni (command: nnictl create --config $CONFIG_PATH)
* Result
  * Linear, MC dropout
    * 66% accuracy in rise and fall
    * of all the right predictions in diretion, 15% close to real value
  * classification 
    * 51% accuracy
    * maybe..problem in model but not fixed (other models are successful)
