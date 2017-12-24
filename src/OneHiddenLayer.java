import java.util.*;

public class OneHiddenLayer extends ANN{

	List<Neuron> hiddenLayer;
	
	int numHiddenNeurons=10;
	double [][] weights;

	/**
	 *
	 * @param trainingData
	 * @param testingData
	 * The constructor will initialize the data structur of the network ie the inLayer/hiddenLayer/outLayer
	 * We choose to initialize the inLayer with a set of zeros and a normalisation of 28*28 (which will be discussed in the report)
	 * However we want initialize the hidden/out layers to make the dynamic initialization of numHiddenNeurons parameter possible
	 */
	public OneHiddenLayer(Map<Input, Output> trainingData, Map<Input, Output> testingData) {
		generator = new Random();
		this.trainingData = trainingData;
		this.testingData = testingData;
		
		// to be completed
		this.inLayer= new ArrayList<>();
		for(int i=0;i<28*28;i++){
			inLayer.add(new InputNeuron(28*28));
			inLayer.get(i).name="inputNeuron"+i;
			inLayer.get(i).feed(0);
		}

		//Creating ArrayList to hold the neuron of hidden/out layers
		this.hiddenLayer= new ArrayList<>();
		this.outLayer= new ArrayList<>();
	}

	/**
	 *
	 * @param in Input item to initialize the 28*28 InputNeuron with the value of each pixel of the 28*28 pixels in the MNSIT image (0-white/254-black)
	 *           we use the feed method of InputNeuron to match the value of pixel with the output of the InputNeuron
	 */
	public void initialiseInputLayer(Input in){
		int numInput=0;
		for(int ligne=0;ligne<in.data.length;ligne++){
			for (int colonne=0;colonne<in.data.length;colonne++){
				inLayer.get(numInput).feed(in.data[ligne][colonne]);
				numInput++;
			}
		}
	}

	/**
	 *
	 * @param numHiddenNeurons is the number of neuron in the hidden layer
	 *                         taking into account this parameter we will initialize the hiddenLayer by choosing the function of activation
	 *                         (turned out that the Sigmoid is the best choice to have a good accuracy of this implementation)
	 *                         We then link the input/hidden/out layers by adding parents/children to each one
	 *                         Finally we use the initWeights method to initialize weights of the network with values randomly choosen in the range of [-1,1]
	 */
	public void initialiseHiddenOutLayer(int numHiddenNeurons){
		for(int i=0;i<numHiddenNeurons;i++){
			hiddenLayer.add(new Neuron(new Sigmoid()));
			hiddenLayer.get(i).name="hiddenNeuron"+i;
			hiddenLayer.get(i).out=0;
		}
		for(int neuronCache=0;neuronCache<numHiddenNeurons;neuronCache++){
			for(int j=0;j<inLayer.size();j++){
				hiddenLayer.get(neuronCache).addParent(inLayer.get(j));
			}
			hiddenLayer.get(neuronCache).initWeights();
		}


		for(int i=0;i<10;i++){
			outLayer.add(new Neuron(new Sigmoid()));
			outLayer.get(i).name="outputNeuron"+i;
			outLayer.get(i).out=0;
		}
		for(int neuronSorite=0;neuronSorite<10;neuronSorite++){
			for(int j=0;j<hiddenLayer.size();j++){
				outLayer.get(neuronSorite).addParent(hiddenLayer.get(j));
			}
			outLayer.get(neuronSorite).initWeights();
		}

		for(int neuronCache=0;neuronCache<numHiddenNeurons;neuronCache++){
			for(int k=0;k<outLayer.size();k++){
				hiddenLayer.get(neuronCache).addChild(outLayer.get(k));
			}
		}
	}


	/**
	 *
	 * @param in input data of the neural network
	 *           The feed forward step of the gradient descent algorithm
	 *           We feed the NN with an input, those values are propagated forward in the network passing from inputLayer to hiddenLayer and then to the output layer
	 *           through the feed() method of every neuron in each layer
	 *
	 * @return finally we get an array of the size of the number of neuron in the outputLayer
	 */
	public Output feed(Input in){
		// to be completed
		initialiseInputLayer(in);
		double[] output = new double[10];

		for(int j=0;j<numHiddenNeurons;j++) {
			hiddenLayer.get(j).feed();
		}
		for (int i=0;i<10;i++){
			outLayer.get(i).feed();
			output[i]=outLayer.get(i).out;
		}

		return new Output(output);
	}

	/**
	 * Given an outLayer we calculate a Matrix that is
	 * every row of this matrix represent the weights of connection between neuron of hiddenLayer and the output corresponding to the number of the row
	 * every column is the weights of connection between neuron of outputLayer and the hidden neuron corresponding to the index of the column
	 * this method will be used to calculate the backpropagation error in the case of neurons of the hiddenLayer
	 * wa made it this way to make it more "visible" for us becaus of the difficult task of working with the maps of Weights/Neurons
	 */
	public void weightsMatrix(){
		weights=new double[outLayer.size()][hiddenLayer.size()];
		for(int i=0;i<outLayer.size();i++){
			for (Map.Entry<Neuron, Double> entry : outLayer.get(i).w.entrySet()) {
				for(int j=0;j<hiddenLayer.size();j++){
					if(hiddenLayer.get(j).name==entry.getKey().name)
					weights[i][j]=entry.getValue();
				}
			}
		}


	}

	/**
	 *
	 * @param nbIterations The number of iteration the network will be trained on the MNSIT 60000 examples
	 *                     STEP 0 : Read data (input/output) => input is the 28*28 values of pixels, output is the REAL value corresponding to the written digit
	 *                     STEP 1 : FeedForwad the input
	 *                     For that we use the feed method of this class to calculate the output of the NN
	 *                     STEP 2 : FeedBackward of the error
	 *                     STEP 2-1 : we get the matrix of weights connecting the output/hidden neuron
	 *                     STEP 2-2 : for each neuron in the hidden layer we calculate the effect of the error (output of neuron and real output)
	 *                     in each of his children. We sum it up (taking into account the weights of every error through the matrix of weights
	 *                     and the derivative of the output of the hidden neuron)
	 *                     STEP 2-3 : we than proceede with propagating the target we just calculate trought the connections between inputLayer neurons and hiddenLayer neurons
	 *                     through the backPropagate method
	 *                     STEP 3 : we propagate the error into the neurons of the outLayer
	 *                     STEP 4 : update of weights of connection between input/hidden/out layers
	 *                     LOOP into every single input of the dataset
	 *                     test the network with the testingData and get its accuracy
	 *                     LOOP into every single number of iteration
	 *                     calculate how much did it take to train and test the network
	 * @return
	 */
	public Map<Integer,Double> train(int nbIterations) {
		// to be completed
		Map<Integer,Double> data=new HashMap<Integer, Double>();
		initialiseHiddenOutLayer(numHiddenNeurons);
		long startTime = System.currentTimeMillis();
		for(int iter=0; iter<nbIterations;iter++){
			int traitement=0;
			for (Map.Entry<Input, Output> d : trainingData.entrySet()) {
					traitement++;
					Input in = d.getKey();
					Output realOutput = d.getValue();
					//Feed Forward
					feed(in);

					//Feed backward
					weightsMatrix();
					for(int hidden=0; hidden<numHiddenNeurons;hidden++){
						double sum=0;
						for(int i=0;i<10;i++){
							sum+=outLayer.get(i).derivative(outLayer.get(i).out)*(realOutput.value[i]-outLayer.get(i).out)*weights[i][hidden];
						}
						hiddenLayer.get(hidden).backPropagate(sum);
					}
					for(int sortie=0; sortie<10;sortie++){
						outLayer.get(sortie).backPropagate(realOutput.value[sortie]);
					}
					if(traitement%100==0)	System.out.println("["+iter+"] - Traitement de l'image "+traitement+" terminé ");
			}
			//A chaque itération i on calcule l'erreur globale dans cette itération SUR LES DONNEES DE TEST
			data.put(iter,test(testingData,iter));
			System.out.println("OneHiddenLayer-------------------FIN ITERATION N°"+iter+"-------------------");
		}
		long endTime = System.currentTimeMillis();
		System.out.println("Temps d'execution :" + (endTime - startTime)/(1000*60) + " minutes");//différence donnée en millisecondes..
		return data;
	}

	

}