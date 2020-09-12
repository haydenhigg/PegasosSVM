module.exports = class {
	constructor(inp, out, options = {}) {
		if ([...new Set(out)].length > 2)
			throw RangeError('too many output options');

		this.inp = inp;
		this.out = out;

		this.lambda = options.lambda || 1;
		this.k = options.k || 1;
		this.weights = options.weights || this.inp[0].map(() => 0);
		this.projection = options.projection === undefined ? true : false;
		this.outputs = options.outputs || [1, -1];

		this.out = this.out.map(o => 1 - 2 * this.outputs.indexOf(o));
	}

	__dot(...args) { // vector multiplication
		return args[0].reduce((s, n, i) => s + (n * args.slice(1, args.length).reduce((aS, an) => aS * an[i], 1)), 0);
	}

	__add(...args) { // vector addition
		return args[0].map((n, i) => n + args.slice(1, args.length).reduce((aS, an) => aS + an[i], 0));
	}

	__train(iters) {
		var w = this.weights;

		for (let t = 0; t <= iters; t++) {
			let i = Math.floor(Math.random() * this.out.length);
			let tInp = this.inp[i];
			let tOut = this.out[i];

			let eta = 1 / (this.lambda * (t + 1));
			let score = this.__dot(w, tInp);
			let d = 1 - eta * this.lambda;

			if (tOut * score < 1) {
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
		var w = this.weights;

		for (let t = 0; t <= iters; t++) {
			let is = [];

			while (is.length < this.k) {
				let i = Math.floor(Math.random() * this.out.length);
				if (is.indexOf(i) < 0)
					is.push(i);
			}

			let limitedIs = is.filter(i => this.__dot(w, this.inp[i]) * this.out[i] < 1);

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
		if (iters < 0 || !Number.isSafeInteger(iters)) return;

		if (this.k === 1)
			this.__train(iters);
		else
			this.__miniBatchTrain(iters);

		this.bias = Math.max.apply(null, this.inp.filter((_, i) => this.out[i] === -1).map(i => this.__dot(i, this.weights)));

		return this;
	}

	predict(x) {
		return ((this.__dot(x, this.weights) - this.bias) <= 0 ? this.outputs[1] : this.outputs[0]);
	}
};