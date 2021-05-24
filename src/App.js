import React, { Component } from "react";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";

import { RPSDataset } from "./common/data.js";
import { getModel } from "./common/models.js";
import { train } from "./common/train.js";
import { doSinglePrediction } from "./common/evaluationHelpers.js";
import Model from "./model.js";
import 'bootstrap/dist/css/bootstrap.min.css';
import './index.css';

const DETECTION_PERIOD = 2000;

class App extends Component {
  state = {
    currentModel: null,
    areDataLoaded: false,
    isModelTrained: false,
    isModelCreated: false,
    webcamActive: false,
    camMessage: "",
    demo: false
  };

  renderModel = () => {
    return this.state.demo && <Model key="demo" />;
  };

  componentDidMount() {
    window.tf = tf;
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  detectWebcam = async () => {
    await this.sleep(100);
    const video = document.querySelectorAll(".captureCam");
    const feedbackCanvas = document.getElementById("compVision");

    if (video[0]) {
      const options = { feedbackCanvas };
      const predictions = await doSinglePrediction(
        this.model,
        video[0],
        options
      );
      const camMessage = predictions
        .map(p => ` ${p.className}: %${(p.probability * 100).toFixed(2)}`)
        .toString();
      this.setState({ camMessage });
      setTimeout(this.detectWebcam, DETECTION_PERIOD);
    }
  };

  render() {
    return (
        <div className="bg align-middle">
          <h1 class="text-white">Model Klasyfikacji Obrazu</h1>
        <div>
          <button class="btn btn-outline-light btn-lg m-2 align-middle" 
            onClick={async () => {
              this.setState({ loadDataMessage: "Loading data" });
              const data = new RPSDataset();
              this.data = data;
              await data.load();
              this.setState({ areDataLoaded: true });
            }}
          >
            Load examples
          </button>
          {this.state.areDataLoaded && (<h3 class="text-white">Data has been loaded!</h3>)}
        </div>
        <div>
          <button class="btn btn-outline-light btn-lg m-2 align-middle" 
            onClick={() => {
              const model = getModel();
              tfvis.show.modelSummary(
                    { name: "Model Architecture" },
                    model
                  );
              this.model = model;
              this.setState({ isModelCreated: true });
            }}
            disabled={!this.state.areDataLoaded}
          >
            Create model
          </button>
          {this.state.isModelCreated && (<h3 class="text-white">Model has been created!!!</h3>)}
        </div>
        <div>
          <button class="btn btn-outline-light btn-lg m-2 align-middle" 
            onClick={async () => {
              if (!this.data || !this.model) return;
              await train(this.model, this.data, 12)
              .then(() => {
                return this.setState({isModelTrained: true})
              })
              .catch(err => {
                console.log(err)
              })
            }}
            disabled={!this.state.isModelCreated}
          >
            Train your model
          </button>
          {this.state.isModelTrained && (<h3 class="text-white">Your model have beed trained!!!</h3>)}
        </div>
        <div>
        <button class="btn btn-outline-light btn-lg m-2 align-middle" 
          onClick={async () => {
            if (!this.model) return;
            await this.model.save("downloads://model");
          }}
          disabled={!this.state.isModelTrained}
        >
          Download
        </button>
        </div>
        <div>
          <button className="btn btn-outline-light btn-lg m-2 align-middle" 
            onClick={() => {
              this.setState(prevState => ({
                webcamActive: false,
                demo: !prevState.demo
              }));
            }}
            disabled={!this.state.isModelTrained}
          >
            {this.state.demo
              ? "Turn off Demo"
              : "Show Demo"}
          </button>
          {this.renderModel()}
        </div>
      </div>
    );
  }
}

export default App;