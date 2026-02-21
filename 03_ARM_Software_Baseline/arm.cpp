#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>

using namespace std;

// ---------------- ReLU ----------------
inline float relu(float x) {
    return x > 0 ? x : 0;
}

// ---------------- Load file ----------------
void load_file(const string &filename, vector<float> &data) {
    ifstream file(filename);
    if(!file.is_open()){
        cout << "ERROR opening " << filename << endl;
        exit(1);
    }
    float val;
    while(file >> val)
        data.push_back(val);
    file.close();
    cout << filename << " loaded. Size = " << data.size() << endl;
}

// ---------------- Conv2D ----------------
void conv2d(vector<float>& input,
            vector<float>& output,
            vector<float>& weight,
            vector<float>& bias,
            int in_ch, int out_ch,
            int size, int k)
{
    int pad = 1;

    for(int oc = 0; oc < out_ch; oc++) {
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {

                float sum = bias[oc];

                for(int ic = 0; ic < in_ch; ic++) {
                    for(int ki = 0; ki < k; ki++) {
                        for(int kj = 0; kj < k; kj++) {

                            int x = i + ki - pad;
                            int y = j + kj - pad;

                            if(x >= 0 && x < size &&
                               y >= 0 && y < size)
                            {
                                int in_idx =
                                    ic*size*size +
                                    x*size + y;

                                int w_idx =
                                    oc*(in_ch*k*k) +
                                    ic*(k*k) +
                                    ki*k + kj;

                                sum += input[in_idx] *
                                       weight[w_idx];

                            }
                        }
                    }
                }

                output[oc*size*size + i*size + j] = relu(sum);
            }
        }
    }
}

// ---------------- MaxPool 2x2 ----------------
void maxpool(vector<float>& input,
             vector<float>& output,
             int ch, int size)
{
    int out_size = size / 2;

    for(int c = 0; c < ch; c++) {
        for(int i = 0; i < out_size; i++) {
            for(int j = 0; j < out_size; j++) {

                float m = -1e9;

                for(int ki = 0; ki < 2; ki++) {
                    for(int kj = 0; kj < 2; kj++) {

                        int x = i*2 + ki;
                        int y = j*2 + kj;

                        int idx =
                            c*size*size +
                            x*size + y;

                        m = max(m, input[idx]);
                    }
                }

                output[c*out_size*out_size +
                       i*out_size + j] = m;
            }
        }
    }
}

// ---------------- Fully Connected ----------------
void fc_layer(vector<float>& input,
              vector<float>& output,
              vector<float>& weight,
              vector<float>& bias,
              int in_f, int out_f)
{
    for(int o = 0; o < out_f; o++) {

        float sum = bias[o];

        for(int i = 0; i < in_f; i++)
            sum += input[i] *
                   weight[o*in_f + i];

        output[o] = sum;
    }
}

// ================= MAIN =================
int main()
{
    vector<float> input;
load_file("image_8.txt", input);

// ---------------- Normalize input ----------------
for(int c=0; c<3; c++){
    for(int i=0; i<32*32; i++){
        int idx = c*32*32 + i;
        input[idx] = 2.0f*input[idx] - 1.0f; // [-1,1] like Python
    }
}

cout << "Image size: " << input.size() << endl;
    // Load weights
    vector<float> c1_w,c1_b,
                  c2_w,c2_b,
                  c3_w,c3_b,
                  fc_w,fc_b;

    load_file("conv1.weight.txt", c1_w);
    load_file("conv1.bias.txt",   c1_b);
    load_file("conv2.weight.txt", c2_w);
    load_file("conv2.bias.txt",   c2_b);
    load_file("conv3.weight.txt", c3_w);
    load_file("conv3.bias.txt",   c3_b);
    load_file("fc.weight.txt",    fc_w);
    load_file("fc.bias.txt",      fc_b);

    vector<float> out1(16*32*32),
                  p1(16*16*16),
                  out2(32*16*16),
                  p2(32*8*8),
                  out3(64*8*8),
                  p3(64*4*4),
                  fc_out(10);

    auto start = chrono::high_resolution_clock::now();

    conv2d(input, out1, c1_w, c1_b, 3, 16, 32, 3);
    maxpool(out1, p1, 16, 32);

    conv2d(p1, out2, c2_w, c2_b, 16, 32, 16, 3);
    maxpool(out2, p2, 32, 16);

    conv2d(p2, out3, c3_w, c3_b, 32, 64, 8, 3);
    maxpool(out3, p3, 64, 8);

    fc_layer(p3, fc_out, fc_w, fc_b,
             64*4*4, 10);

    auto end = chrono::high_resolution_clock::now();

    int pred = max_element(
        fc_out.begin(),
        fc_out.end()
    ) - fc_out.begin();

    cout << "Predicted class: "
         << pred << endl;

    cout << "Inference time: "
         << chrono::duration<double, milli>(end-start).count()
         << " ms" << endl;

    return 0;
}
