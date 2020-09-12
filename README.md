# PegasosSVM

A simple, lightweight package for linear classification based off the support vector machine as described in [this study.](https://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf)

## Creating a model

```js
const PegasosSVM = require('pegasossvm');

const inputValues = [[...], [...], ...]; // input array with each item being its own n-dimensional feature array
const outputValues = [...];

const options = { // the completely optional additional parameters, shown here at their defaults
  lambda: 1, // regularization parameter
  k: 1, // change if you want to use mini-batch learning in each iteration
  weights: [0, ..., 0], // initial weights, start at 0 by default
  projection: true, // use a projection step; this was shown to have little effect but is offered anyway
  outputs: [1, -1] // if you want to use output values besides -1 and 1 in training and predicting, set this to [yourValue1, yourValue2]
}

const model = new PegasosSVM(inputValues, outputValues, options);

model.train(iterations);
console.log(model.predict(testValue)); //=> predicted output
```

## Notes

- As the algorithm is intended to decrease the learning rate as the training progresses, so a regularization parameter higher than other Perceptrons' learning rate is often helpful (the ideal value seemed to be around 1 for many of the training sets I tested)
- The algorithm selects random batches of input/output pairs to train on in each iteration, and if it selects unrepresentative pairs within the first few iterations (when the learning rate is still high) then it will have very poor performance
- This implementation calculates a bias to be used when predicting. If you wish to override or view this, it is available as the value `model.bias` and is set at the end of the `train` method.
- If k == 1 then the special case algorithm is used, otherwise the general algorithm for a mini-batch of size k is used.
- The `train` method returns its instance, so method calls can be chained.
