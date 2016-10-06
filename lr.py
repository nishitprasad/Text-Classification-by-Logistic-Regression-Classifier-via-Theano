import numpy as np
import theano
import theano.tensor as T

class LogisticRegression():
    
    def LR(self, train_x, train_y, test_x, test_y):
        n_classes = np.unique(train_y).shape[0]
        n_instances = train_x.shape[0]
        n_feats = train_x.shape[1]
        n_epoches = 2000
        
        # declare Theano symbolic variables
        x = T.matrix("x")
        y = T.ivector("y")
        w = theano.shared(np.random.randn(n_feats,n_classes), name="w")
        b = theano.shared(np.zeros(n_classes), name="b")
        
        print("Initial model:")
        print(w.get_value())
        print(b.get_value())

        # construct Theano expression graph
        p_y_given_x = T.nnet.softmax(T.dot(x, w) + b)
        xent = -T.mean(T.log(p_y_given_x)[T.arange(n_instances), y])
        cost = xent + 0.01 * (w ** 2).sum()       # The cost to minimize
        gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
        y_pred = T.argmax(p_y_given_x, axis=1)
        error = T.mean(T.neq(y_pred, y))
        
        # compile
        train = theano.function(inputs=[x,y],
              outputs=[error, cost],
              updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
        
        # train
        for i in range(n_epoches):
            error, cost = train(train_x, train_y)
            
        print 'Final Current error: %.4f | Current cost: %.4f' % (error, cost)
        
        #test
        test = theano.function(inputs=[x],
                               outputs = y_pred)
        
        print("Final model:")
        print(w.get_value())
        print(b.get_value())

        pred_y = test(test_x)
        
        #Calculate True Positive, True Negative, False Positive and False Negative Values
        tn = 0	#True Negative
        tp = 0	#True Positive
        fn = 0	#False Negative
        fp = 0	#False Positive
        for t_i in range(len(test_y)):
            if(test_y[t_i]==pred_y[t_i]):
                if(test_y[t_i]==0 and pred_y[t_i]==0):
                    tn += 1
                elif(test_y[t_i]==1 and pred_y[t_i]==1):
                    tp += 1
            elif(test_y[t_i]==1 and pred_y[t_i] == 0):
                fn += 1
            elif(test_y[t_i]==0 and pred_y[t_i]==1):
                fp += 1
        
        #Calculate the precision, recall and accuracy values
        precision = float(tp)/float(tp+fp)
        recall = float(tp)/float(tp+fn)
        accuracy = float(tp+tn)/float(tp+tn+fp+fn)
        
		#Calculate the F1 Score
        f1_score = 2*precision*recall/(precision+recall)
        print "Precision: ", precision
        print "Recall: ", recall
        print "Accuracy: ", accuracy
        print "F1 score: ", f1_score
                    