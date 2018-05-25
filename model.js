
export class Model{
    init(){
        throw new Error("Abstract method must be implemented in the derived class");
    }
    predict(inputXs){
        throw new Error("Abstract method must be implemented in the derived class");
    }
    train(inputXs,inputYs){
        throw new Error("Abstract method must be implemented in the derived class");
    }
    fit(inputXs,inputYs,iterationCount = 100){
        for(let i = 0;i<iterationCount;i++){
            this.train(inputXs,inputYs);
        }
    }
    loss(predictedYs,labels){
        throw new Error("Abstract method must be implemented in the derived class");
    }
}