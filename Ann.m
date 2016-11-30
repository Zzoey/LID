%Author : Vinay Chandragiri
%Clean-up
 clc;

%Normalize the Input,Checking whether the Inputs needs to be normalized or not
function BackPropAlgo(Input, Output)
	if max(abs(Input(:)))> 1
		Norm_Input = Input / max(abs(Input(:))); %Need to normalize
		else
		Norm_Input = Input;
	end

	if max(abs(Output(:))) >1 %Checking Whether the Outputs needs to be normalized or not
		Norm_Output = Output / max(abs(Output(:)));
		else
		Norm_Output = Output;
	end

	m = 2; %Number of hidden neurons in hidden layer
	[l,b] = size(Input);  %Size of Input and Output Vectors
	[n,a] = size(Output);

	%Initialize the weight matrices with random weights
	V = rand(l,m); % Weight matrix from Input to Hidden
	W = rand(m,n); % Weight matrix from Hidden to Output

	%Setting count to zero, to know the number of iterations
	count = 0;
	[errorValue delta_V delta_W] = trainNeuralNet(Norm_Input,Norm_Output,V,W); %Training the neural network

	%Checking if error value is greater than 0.1. If yes, we need to train the network again. User can decide the threshold value
	while errorValue > 0.05
		count = count + 1;
		Error_Mat(count)=errorValue; %Store the error value into a matrix to plot the graph
		W=W+delta_W; %Change the weight metrix V and W by adding delta values to them
		V=V+delta_V;
		count; %Calling the function with another overload.Now we have delta values as well.
		[errorValue delta_V delta_W]=trainNeuralNet(Norm_Input,Norm_Output,V,W,delta_V,delta_W);
	end

	%This code will be executed when the error value is less than 0.1
	if errorValue < 0.05
		count=count+1;
		Error_Mat(count)=errorValue;
	end

	%Calculating error rate
	Error_Rate=sum(Error_Mat)/count;
	figure;
	%setting y value for plotting graph
	y=[1:count];
	plot(y, Error_Mat);

end


function [errorValue delta_V delta_W] = trainNeuralNet(Input, Output, V, W, delta_V, delta_W)
	Output_of_InputLayer = Input; %Output of Input Layer is same as the Input of Input  Layer

	%Calculating Input of the Hidden Layer.Here we need to multiply the Output of the Input Layer with the -synaptic weight. That weight is in the matrix V.
	Input_of_HiddenLayer = V' * Output_of_InputLayer;
	[m n] = size(Input_of_HiddenLayer); %Calculate the size of Input to Hidden Layer
	Output_of_HiddenLayer = 1./(1+exp(-Input_of_HiddenLayer)); %Sigmoidal Function calculate the Output of the Hidden Layer
	Input_of_OutputLayer = W'*Output_of_HiddenLayer;
	clear m n;

	[m n] = size(Input_of_OutputLayer);
	Output_of_OutputLayer = 1./(1+exp(-Input_of_OutputLayer));
	difference = Output - Output_of_OutputLayer; %Error using Root Mean Square method
	square = difference.*difference;
	errorValue = sqrt(sum(square(:)));
	clear m n;
	[n a] = size(Output);

	%Calculate the matrix 'd' with respect to the desired output
	for i = 1 : n
		for j = 1 : a
		d(i,j) =(Output(i,j)-Output_of_OutputLayer(i,j))*Output_of_OutputLayer(i,j)*(1-Output_of_OutputLayer(i,j));
		end
	end

	%Now, calculate the Y matrix
	Y = Output_of_HiddenLayer * d;

	%Checking number of arguments. We are using function overloading On the first iteration, we don't have delta V and delta W So we have to initialize with zero. The size of delta V and delta W will be same as that of V and W matrix respectively (nargin - no of arguments)
	if nargin == 4
 		delta_W=zeros(size(W));
		delta_V=zeros(size(V));
	end

	etta=0.6;alpha=1; %Initializing etta with 0.6 and alpha with 1
	%Calculating delta W
	delta_W= alpha.*delta_W + etta.*Y;
	error = W*d; %Calculating error matrix
	clear m n; %Calculating d_star
	[m n] = size(error);
	for i = 1 : m
		for j = 1 :n
		 	d_star(i,j)= error(i,j)*Output_of_HiddenLayer(i,j)*(1-Output_of_HiddenLayer(i,j));
		end
	end

	X = Input * d_star'; % Matrix, X (Input * transpose of d_star)
	%Calculating delta V
	delta_V=alpha*delta_V+etta*X;

end
