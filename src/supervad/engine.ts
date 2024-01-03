import * as ort from 'onnxruntime-web';

export class SuperVADEngine {

    static async create() {
        const session = await ort.InferenceSession.create('/supervad.onnx');
        return new SuperVADEngine(session);
    }

    static readonly TOKEN_SIZE = 320;

    private _buffer = new Float32Array(3200);
    private _session: ort.InferenceSession;

    constructor(session: ort.InferenceSession) {
        this._session = session;
    }

    async predict(token: Float32Array) {
        if (token.length !== SuperVADEngine.TOKEN_SIZE) {
            throw new Error('Invalid token length, expected ' + SuperVADEngine.TOKEN_SIZE + ' items, got: ' + token.length);
        }

        // Shift biffer
        for (let i = 0; i < 3200 - 320; i++) {
            this._buffer[i] = this._buffer[i + 320];
        }

        // Load token
        for (let i = 0; i < 320; i++) {
            this._buffer[3200 - 320 + i] = token[i];
        }

        // Run inference
        const input = new ort.Tensor('float32', this._buffer, [1, this._buffer.length]);
        const results = await this._session.run({ input });

        // Return probability
        return results.output.data[0] as number;
    }
}

export class SuperVADStreamEngine {

    static async create(args: {
        activation_threshold: number,
        activation_tokens: number,
        deactivation_threshold: number,
        deactivation_tokens: number,
        prebuffer_tokens: number,
    }) {
        const engine = await SuperVADEngine.create();
        return new SuperVADStreamEngine({ ...args, engine });
    }

    static readonly TOKEN_SIZE = 320;

    readonly activation_threshold: number;
    readonly activation_tokens: number;
    readonly deactivation_threshold: number;
    readonly deactivation_tokens: number;
    readonly prebuffer_tokens: number;
    private readonly engine: SuperVADEngine;
    private state: 'active' | 'activating' | 'deactivated' | 'deactivating';
    private preBuffer = Float32Array.from([]);
    private activeBuffer = Float32Array.from([]);
    private stateTokens = 0;

    constructor(args: {
        engine: SuperVADEngine,
        activation_threshold: number,
        activation_tokens: number,
        deactivation_threshold: number,
        deactivation_tokens: number,
        prebuffer_tokens: number,
    }) {
        this.engine = args.engine;
        this.activation_threshold = args.activation_threshold;
        this.activation_tokens = args.activation_tokens;
        this.deactivation_threshold = args.deactivation_threshold;
        this.deactivation_tokens = args.deactivation_tokens;
        this.prebuffer_tokens = args.prebuffer_tokens;
        this.state = 'deactivated';
    }

    async process(token: Float32Array): Promise<{ state: 'activating' } | { state: 'active' } | { state: 'activation-canceled' } | { state: 'deactivating' } | { state: 'deactivation-canceled' } | { state: 'complete', buffer: Float32Array } | { state: 'unchanged' }> {

        //
        // Perform VAD
        //

        const prediction = await this.engine.predict(token);
        console.log(prediction);

        //
        // Append to token buffer
        //

        this.preBuffer = concat(this.preBuffer, token);
        if (this.preBuffer.length > this.prebuffer_tokens * SuperVADEngine.TOKEN_SIZE) {
            this.preBuffer = this.preBuffer.subarray(this.preBuffer.length - this.prebuffer_tokens * SuperVADEngine.TOKEN_SIZE);
        }

        //
        // Handle activation
        //

        if (this.state === 'deactivated') {
            if (prediction >= this.activation_threshold) {
                if (this.activation_tokens <= 1) {
                    this.state = 'active';
                    this.activeBuffer = this.preBuffer;
                    return { state: 'active' };
                } else {
                    this.state = 'activating';
                    this.activeBuffer = this.preBuffer;
                    this.stateTokens = 1;
                    return { state: 'activating' };
                }
            }
        }

        if (this.state === 'activating') {
            this.activeBuffer = concat(this.activeBuffer, token);

            // Update counters
            if (prediction >= this.activation_threshold) {
                this.stateTokens++;
            } else {
                this.stateTokens--;
            }

            // Activation failed
            if (this.stateTokens <= 0) {
                this.state = 'deactivated';
                this.stateTokens = 0;
                this.activeBuffer = new Float32Array(0);
                return { state: 'activation-canceled' };
            }

            // Activation succeeded
            if (this.stateTokens >= this.activation_tokens) {
                this.state = 'active';
                return { state: 'active' };
            }
        }

        // Handle active state
        if (this.state === 'active') {
            this.activeBuffer = concat(this.activeBuffer, token);

            // Check if we should start deactivation
            if (prediction <= this.deactivation_threshold) {
                this.state = 'deactivating';
                this.stateTokens = 1;
                return { state: 'activating' };
            }
        }

        // Handle deactivation
        if (this.state === 'deactivating') {
            this.activeBuffer = concat(this.activeBuffer, token);

            // If voice activity detected
            if (prediction >= this.activation_threshold) {
                this.state = 'active';
                this.stateTokens = 0;
                return { state: 'deactivation-canceled' };
            }

            // Update coutners
            if (prediction <= this.deactivation_threshold) {
                this.stateTokens++;
                console.warn(this.stateTokens);
            } else {
                this.stateTokens--;
                console.warn(this.stateTokens);
            }

            // If deactivation failed
            if (this.stateTokens <= 0) {
                this.state = 'active';
                this.stateTokens = 0;
                return { state: 'deactivation-canceled' };
            }

            // If deactivation successful
            if (this.stateTokens >= this.deactivation_tokens) {
                const buffer = this.activeBuffer;
                this.activeBuffer = new Float32Array(0);
                this.state = 'deactivated';
                this.stateTokens = 0;
                return { state: 'complete', buffer };
            }
        }

        return { state: 'unchanged' };
    }
}

function concat(a: Float32Array, b: Float32Array) {
    const c = new Float32Array(a.length + b.length);
    c.set(a);
    c.set(b, a.length);
    return c;
}