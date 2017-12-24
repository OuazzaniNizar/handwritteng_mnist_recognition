import javax.sound.sampled.Line;
import java.util.*;

public class Neuron {
	/** name of the neuron */
	String name;	
	/** List of parent neurons, i.e. the list of neurons that are used as input */
	List<Neuron> parents;
	/** list of child neurons,i.e. the list of neurons that use the output of this neuron */
	List<Neuron> children;
	/** activation function */
	Activation h;
	/** weight of the input neurons: it maps an input neuron to a weight */
	Map<Neuron,Double> w;
	/** value of the learning rate */
	final static double eta = 0.1;
	/** current value of the output of the neuron */
	protected double out;
	/** current value of the error of the neuron */
	protected double error;
	/** random number generator */
	public static Random generator;
	//Success! Your submission appears on this page. The submission confirmation number is b93186b2-6478-4d76-8372-1d107fb25937. Copy and save this number as proof of your submission.

	/** 
	 * returns the current value of the error for that neuron.
	 * @return
	 */
	public double getError(){return error;}
	
	/**
	 * Constructor
	 * @param h is an activation function (linear, sigmoid, tanh)
	 */
	public Neuron(Activation h){
		if (generator==null)
			generator = new Random();
		this.h=h;
		parents = new ArrayList<>();
		children = new ArrayList<>();	
		w = new LinkedHashMap<>();
	}
	
	public void addParent(Neuron parent){
		parents.add(parent);
	}
	
	public void addChild(Neuron child){
		children.add(child);
	}
	
	/**
	 * Initializes randomly the weights of the incoming edges 
	 */
	public void initWeights(){
		// to be completed
		for(Neuron n : parents ){
			//we create a random weight associated to the input neuron in a range of [-1,1]
			w.put(n,(2*generator.nextDouble())-1);
		}
	}
	
	/**
	 * Computes the output of a neuron that is either in the hidden layer or in the output layer. 
	 * (there are no arguments as the neuron is not an inputNeuron)
	 */
	public void feed(){
		// to be completed
		double sum=0;
		for(Neuron n : parents){
			//sum=sum+Weighti*Inputi
			sum+=w.get(n)*n.out;
		}
		out=h.activate(sum/(1));
	}


	/**
	 *
	 * @param x take an output and calculate its derivative
	 *          this method will be used in the backpropagation formula
	 * @return
	 */
	public double derivative(double x){
		if(h instanceof Linear) return 1;
		if(h instanceof Tanh)return (1-(x*x));
		else return (x*(1-x));
	}

	/**
	 * back propagation for a neuron in the output layer 
	 * @param target is the correct value.
	 */
	public void backPropagate(double target){

		//Check if the neuron using this method is a hidden neuron
		//if yes, back propagate the error through its connections with the input neuron
		//if no, the neuron is then an output neuron, we will the backpropagate the error withing its predecessors which
		//are the hidden neurons in the case of OneHiddenLayer or the inputLayer in the case of SingleLyaer
		//for both the formula used is the gradient descent formula
		if (children.size() != 0) {
			for (Map.Entry<Neuron, Double> entry : w.entrySet()) {
				double weight = entry.getValue();
				double delta = (target) * derivative(out);
				double gradient = entry.getKey().eta * delta * (entry.getKey().out);
				weight += gradient;
				w.put(entry.getKey(), weight);
			}
		}else {
			for (Map.Entry<Neuron, Double> entry : w.entrySet()) {
				double weight = entry.getValue();
				double delta = (target - out) * derivative(out);
				double gradient = entry.getKey().eta * delta * (entry.getKey().out);
				weight += gradient;
				w.put(entry.getKey(), weight);
			}
		}
	}


	/**
	 * returns the current ouput (it should be called once the output has been computed, 
	 * i.e. after calling feed)
	 * @return the current value of the ouput
	 */
	public double getCurrentOutput(){
		return out;
	}
	
	/** returns the name of the neuron *
	 * 
	 */
	public String toString(){
		return name + " out: " + out ;
	}
	

}