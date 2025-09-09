use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;


pub struct SplitLayer;

impl Layer for SplitLayer {
    fn graph(
        input_shapes: &Vec<&Vec<usize>>,
        input_types: &Vec<DatumType>,
        constants: &Vec<Option<(&ndarray::ArrayD<f32>, DatumType)>>, // adjust type if you use Fr
        attributes: &Vec<&AttributeProto>,
    ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
        let mut graph = Graph::new();

        // ---- axis (always attribute) ----
        let axis_attr = attributes.iter().find(|x| x.name == "axis").unwrap();
        let axis: isize = axis_attr.i as isize;
        let axis = if axis < 0 {
            input_shapes[0].len() as isize + axis
        } else {
            axis
        } as usize;

        // ---- split: opset 11 vs opset 13+ ----
        let split: Vec<usize> = if let Some(attr) = attributes.iter().find(|x| x.name == "split") {
            // ✅ Opset 11: split is attribute
            attr.ints.iter().map(|x| *x as usize).collect()
        } else if input_shapes.len() > 1 {
            // ✅ Opset 13+: split is the 2nd input (tensor)
            if let Some((arr, _dtype)) = &constants[1] {
                arr.iter().map(|x| *x as usize).collect()
            } else {
                panic!("Split tensor not constant, dynamic splits not supported yet");
            }
        } else {
            // ✅ Default: equal split (if split not provided)
            let dim = input_shapes[0][axis];
            let num_outputs = 2; // <-- adjust if your framework knows output count
            vec![dim / num_outputs; num_outputs]
        };

        // ---- Output shapes ----
        let mut output_shapes = vec![];
        for i in 0..split.len() {
            let mut output_shape = input_shapes[0].clone();
            output_shape[axis] = split[i];
            output_shapes.push(output_shape);
        }

        // ---- Build graph ----
        if axis == input_shapes[0].len() - 1 {
            // special case with permutation
            let n = input_shapes[0].len();
            let (mut a, mut b) = (input_shapes[0][n - 2], input_shapes[0][n - 1]);
            (a, b) = (
                util::next_pow(a as u32) as usize,
                util::next_pow(b as u32) as usize,
            );

            let permutation = (
                (0..b).map(|x| x * a).collect(),
                (0..a).map(|x| x).collect(),
            );

            let permute = graph.addBB(Box::new(RepeaterBasicBlock {
                basic_block: Box::new(PermuteBasicBlock {
                    permutation: permutation,
                    n: a,
                    m: b,
                }),
                N: 2,
            }));

            let split_bb = graph.addBB(Box::new(SplitBasicBlock {
                axis: (axis - 1) as usize,
                split: split.clone(),
            }));

            let mut permute_backs = vec![];
            for i in 0..split.len() {
                let (mut a, mut b) = (output_shapes[i][n - 2], output_shapes[i][n - 1]);
                (a, b) = (
                    util::next_pow(a as u32) as usize,
                    util::next_pow(b as u32) as usize,
                );
                let permutation_back = (
                    (0..a).map(|x| x * b).collect(),
                    (0..b).collect(),
                );
                let permute_back = graph.addBB(Box::new(RepeaterBasicBlock {
                    basic_block: Box::new(PermuteBasicBlock {
                        permutation: permutation_back,
                        n: b,
                        m: a,
                    }),
                    N: 2,
                }));
                permute_backs.push(permute_back);
            }

            let permute_output = graph.addNode(permute, vec![(-1, 0)]);
            let split_output = graph.addNode(split_bb, vec![(permute_output, 0)]);
            for i in 0..split.len() {
                let output = graph.addNode(permute_backs[i], vec![(split_output, i)]);
                graph.outputs.push((output, 0));
            }
        } else {
            let split_bb = graph.addBB(Box::new(SplitBasicBlock {
                axis: axis as usize,
                split: split.clone(),
            }));
            let split_output = graph.addNode(split_bb, vec![(-1, 0)]);
            for i in 0..split.len() {
                graph.outputs.push((split_output, i));
            }
        }

        let num_outputs = output_shapes.len();
        (graph, output_shapes, vec![input_types[0]; num_outputs])
    }
}


// pub struct SplitLayer;
// impl Layer for SplitLayer {
//   fn graph(
//     input_shapes: &Vec<&Vec<usize>>,
//     input_types: &Vec<DatumType>,
//     _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
//     attributes: &Vec<&AttributeProto>,
//   ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
//     let mut graph = Graph::new();

//     // This code is for Opset 11; in the latest version of ONNX, "split" is in inputs instead of the attributes
//     let axis: isize = attributes.iter().filter(|x| x.name == "axis").next().unwrap().i as isize;
//     let axis = (if axis < 0 { input_shapes[0].len() as isize + axis } else { axis }) as usize;
//     let split = attributes.iter().filter(|x| x.name == "split").next().unwrap().ints.iter().map(|x| *x as usize).collect::<Vec<usize>>();

//     let mut outputShapes = vec![];
//     for i in 0..split.len() {
//       let mut outputShape = input_shapes[0].clone();
//       outputShape[axis] = split[i];
//       outputShapes.push(outputShape);
//     }

//     if axis == input_shapes[0].len() - 1 {
//       // permute inputs
//       let n = input_shapes[0].len();
//       let mut a = input_shapes[0][n - 2];
//       let mut b = input_shapes[0][n - 1];
//       (a, b) = (util::next_pow(a as u32) as usize, util::next_pow(b as u32) as usize);
//       let permutation = ((0..b).map(|x| x * a).collect(), (0..a).map(|x| x).collect());
//       let permute = graph.addBB(Box::new(RepeaterBasicBlock {
//         basic_block: Box::new(PermuteBasicBlock {
//           permutation: permutation,
//           n: a,
//           m: b,
//         }),
//         N: 2,
//       }));
//       let split_bb = graph.addBB(Box::new(SplitBasicBlock {
//         axis: (axis - 1) as usize,
//         split: split.clone(),
//       }));
//       let mut permute_backs = vec![];
//       for i in 0..split.len() {
//         let (mut a, mut b) = (outputShapes[i][n - 2], outputShapes[i][n - 1]);
//         (a, b) = (util::next_pow(a as u32) as usize, util::next_pow(b as u32) as usize);
//         let permutation_back = ((0..a).map(|x| x * b).collect(), (0..b).collect());
//         let permute_back = graph.addBB(Box::new(RepeaterBasicBlock {
//           basic_block: Box::new(PermuteBasicBlock {
//             permutation: permutation_back,
//             n: b,
//             m: a,
//           }),
//           N: 2,
//         }));
//         permute_backs.push(permute_back);
//       }

//       let permute_output = graph.addNode(permute, vec![(-1, 0)]);
//       let split_output = graph.addNode(split_bb, vec![(permute_output, 0)]);
//       for i in 0..split.len() {
//         let output = graph.addNode(permute_backs[i], vec![(split_output, i)]);
//         graph.outputs.push((output, 0));
//       }
//     } else {
//       let split_bb = graph.addBB(Box::new(SplitBasicBlock {
//         axis: axis as usize,
//         split: split.clone(),
//       }));
//       let split_output = graph.addNode(split_bb, vec![(-1, 0)]);
//       for i in 0..split.len() {
//         graph.outputs.push((split_output, i));
//       }
//     }

//     let num_outputs = outputShapes.len();
//     (graph, outputShapes, vec![input_types[0]; num_outputs])
//   }
// }
