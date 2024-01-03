import { AsyncLock } from "./lock";

export class AsyncLazy<T> {

    private factory: () => T | Promise<T>;
    private lock = new AsyncLock();
    private instance: T | null = null;

    constructor(factory: () => T | Promise<T>) {
        this.factory = factory;
    }

    async get() {

        // Fast check
        if (this.instance) {
            return this.instance!;
        }

        return this.lock.inLock(async () => {
            if (this.instance) {
                return this.instance!;
            }
            this.instance = await this.factory();
            return this.instance!;
        });
    }
}