export class MnistData {

    async load() {

        const MNIST_IMAGES_SPRITE =
            'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';

        const MNIST_LABELS =
            'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

        const img = new Image();

        img.crossOrigin = '';

        img.src = MNIST_IMAGES_SPRITE;

        await new Promise(resolve => {
            img.onload = resolve;
        });

        const canvas = document.createElement('canvas');

        canvas.width = img.width;
        canvas.height = img.height;

        const ctx = canvas.getContext('2d');

        ctx.drawImage(img, 0, 0);

        const imageData =
            ctx.getImageData(
                0,
                0,
                canvas.width,
                canvas.height
            );

        this.images = new Float32Array(
            imageData.data.length / 4
        );

        for (let i = 0; i < this.images.length; i++) {

            this.images[i] =
                imageData.data[i * 4] / 255;
        }

        const response = await fetch(MNIST_LABELS);

        const buffer =
            await response.arrayBuffer();

        this.labels =
            new Uint8Array(buffer);

        this.numExamples =
            this.labels.length;

        console.log(
            "Loaded",
            this.numExamples,
            "examples"
        );
    }


    nextTrainBatch(batchSize) {

        const xs = [];
        const labels = [];

        for (let i = 0; i < batchSize; i++) {

            const index =
                Math.floor(
                    Math.random() *
                    this.numExamples
                );

            const image =
                this.images.slice(
                    index * 28 * 28,
                    (index + 1) * 28 * 28
                );

            xs.push(image);

            const label =
                this.labels[index];

            const oneHot =
                new Array(10).fill(0);

            oneHot[label] = 1;

            labels.push(oneHot);
        }

        return {

            xs: tf.tensor2d(
                xs,
                [batchSize, 28 * 28]
            ),

            labels: tf.tensor2d(
                labels,
                [batchSize, 10]
            )
        };
    }
}