module.exports = class {
	__scale(vec, n = 1) { // scale vector to have L2 norm = n
		let norm = Math.sqrt(vec.reduce((s, i) => s + (i ** 2), 0));

		return vec.map(i => i * (n / norm));
	}
	
	__findOutputs() {
		this.outputs = [...new Set(this.out)];

		if (this.outputs.length > 2)
	 		throw new RangeError('too many unique output possibilities');

		this.out = this.out.map(o => 1 - 2 * this.outputs.indexOf(o)); // map outputs to -1 or 1
        }
	
	constructor(inp, out, options = {}) {
		this.inp = inp.map(this.__scale); // scale to have L2 norm = 1, doesn't change relative positions in solution space
		this.out = out;

		this.lambda = options.lambda || 1;
		this.k = options.k || 1;
		this.weights = options.weights || this.inp[0].map(() => 0);
		this.projection = options.projection === undefined ? true : false;
	}

	__dot(...args) { // vector multiplication
		return args[0].reduce((s, n, i) => s + (n * args.slice(1, args.length).reduce((aS, an) => aS * an[i], 1)), 0);
	}

	__add(...args) { // vector addition
		return args[0].map((n, i) => n + args.slice(1, args.length).reduce((aS, an) => aS + an[i], 0));
	}

	__train(iters) {
		this.__findOutputs();
		
		var w = [...this.weights];

		for (let t = 0; t <= iters; t++) {
			let i = Math.floor(Math.random() * this.out.length);
			let tInp = this.inp[i];
			let tOut = this.out[i];

			let eta = 1 / (this.lambda * (t + 1));
			let score = this.__dot(w, tInp);
			let d = 1 - eta * this.lambda;

			if (tOut * score < 1) { // if y(w · x) < 1
				let dx = eta * tOut;
				let xFactor = tInp.map(xi => xi * dx);

				w = this.__add(w.map(wi => wi * d), xFactor);
			} else
				w = w.map(wi => wi * d)
			
			if (this.projection) {
				let projection = (1 / Math.sqrt(this.lambda)) / Math.sqrt(w.reduce((s, n) => s + (n ** 2)));
				if (projection < 1)
					w = w.map(wi => wi * projection);
			}
		}

		this.weights = w;
	}

	__miniBatchTrain(iters) {
		this.__findOutputs();
		
		let w = [...this.weights];

		for (let t = 0; t <= iters; t++) {
			let is = [];

			// generates set of k distinct indices in the data
			while (is.length < this.k) {
				let i = Math.floor(Math.random() * this.out.length);
				if (is.indexOf(i) < 0)
					is.push(i);
			}

			let limitedIs = is.filter(i => this.__dot(w, this.inp[i]) * this.out[i] < 1); // finds all indices where y(w · x) < 1

			if (limitedIs.length > 0) {
				let eta = 1 / (this.lambda * (t + 1));
				let d = 1 - eta * this.lambda;

				let batchDx = eta / this.k;
				let batchXFactors = this.__add.apply(null, limitedIs.map(i => this.inp[i].map(j => j * this.out[i])));

				w = this.__add(w.map(wi => wi * d), batchXFactors.map(i => i * batchDx));

				if (this.projection) {
					let projection = (1 / Math.sqrt(this.lambda)) / Math.sqrt(w.reduce((s, n) => s + (n ** 2)));
					if (projection < 1)
						w = w.map(wi => wi * projection);
				}
			}
		}

		this.weights = w;
	}

	train(iters = 1) {
		if (iters < 0 || !Number.isSafeInteger(iters)) return; // iters > 0 and iters is an integer

		if (this.k === 1) // use special case of k = 1 for higher performance if possible
			this.__train(iters);
		else
			this.__miniBatchTrain(iters);

		// can try to calculate bias like:
		// b = -(max {x · w | x ∈ inputs && f(w, x) = -1})
		// this.bias = -Math.max.apply(null, this.inp.filter((_, i) => this.out[i] === -1).map(i => this.__dot(i, this.weights)));

		return this;
	}

	predict(x) {
		if (this.outputs.length === 0)
			return null;
		else if (this.outputs.length === 1)
			return this.outputs[0];
		else
			return this.__dot(this.weights, x) > 0 ? this.outputs[0] : this.outputs[1];
	}
};
