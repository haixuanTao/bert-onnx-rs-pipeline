mod utils;
use csv::ReaderBuilder;
use csv::WriterBuilder;
use onnxruntime::*;
use std::fs::File;
use std::time::Instant;
use tokenizers::models::wordpiece::WordPieceBuilder;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::tokenizer::AddedToken;
use tokenizers::tokenizer::{EncodeInput, Encoding, Tokenizer};
use tokenizers::utils::padding::{PaddingDirection::Right, PaddingParams, PaddingStrategy::Fixed};
use tokenizers::utils::truncation::TruncationParams;
use tokenizers::utils::truncation::TruncationStrategy::LongestFirst;
fn main() -> std::result::Result<(), OrtError> {
    let start = Instant::now();
    let vocab_path = "./src/vocab.txt";
    let wp_builder = WordPieceBuilder::new()
        .files(vocab_path.into())
        .continuing_subword_prefix("##".into())
        .max_input_chars_per_word(100)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();

    let mut tokenizer = Tokenizer::new(Box::new(wp_builder));
    tokenizer.with_pre_tokenizer(Box::new(BertPreTokenizer));
    tokenizer.with_truncation(Some(TruncationParams {
        max_length: 60,
        strategy: LongestFirst,
        stride: 0,
    }));
    tokenizer.with_post_processor(Box::new(BertProcessing::new(
        ("[SEP]".into(), 102),
        ("[CLS]".into(), 101),
    )));
    tokenizer.with_normalizer(Box::new(BertNormalizer::new(true, true, false, false)));
    tokenizer.add_special_tokens(&[
        AddedToken {
            content: "[PAD]".into(),
            single_word: false,
            lstrip: false,
            rstrip: false,
        },
        AddedToken {
            content: "[CLS]".into(),
            single_word: false,
            lstrip: false,
            rstrip: false,
        },
        AddedToken {
            content: "[SEP]".into(),
            single_word: false,
            lstrip: false,
            rstrip: false,
        },
        AddedToken {
            content: "[MASK]".into(),
            single_word: false,
            lstrip: false,
            rstrip: false,
        },
    ]);
    tokenizer.with_padding(Some(PaddingParams {
        strategy: Fixed(60),
        direction: Right,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".into(),
    }));

    let mut rdr = ReaderBuilder::new()
        .from_path("./medium.csv")
        .unwrap();
    let environment = environment::Environment::builder()
        .with_name("test")
        .build()?;
    let session = std::sync::Arc::new(std::sync::Mutex::new(
        environment
            .new_session_builder()?
            .use_cuda(0)?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_model_from_file("./src/onnx_model.onnx")?,
    ));
    let setup = Instant::now();

    let df: Vec<EncodeInput> = rdr
        .deserialize()
        .into_iter()
        .map(
            |result: std::result::Result<utils::DataFrame, csv::Error>| -> EncodeInput {
                match result {
                    Ok(rec) => {
                        EncodeInput::Single(rec.Title.unwrap() + " " + &rec.BodyMarkdown.unwrap())
                    }
                    Err(_) => EncodeInput::Single("[ERROR]".to_string()),
                }
            },
        )
        .collect();

    let read = Instant::now();

    let input_ids = tokenizer.encode_batch(df, true).unwrap();

    let mut masks: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>> =
        Vec::new();

    let mut tokens: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>> =
        Vec::new();
    let mut mask = ndarray::Array::zeros((256, 60));
    let mut token = ndarray::Array::zeros((256, 60));
    for (i, input) in input_ids.iter().rev().enumerate() {
        for (j, attention) in input.get_attention_mask().iter().enumerate() {
            mask[[255 - i % 256, j]] = *attention as i64;
        }
        for (j, attention) in input.get_ids().iter().enumerate() {
            token[[255 - i % 256, j]] = *attention as i64;
        }
        if (i + 1) % 256 == 0 || i == input_ids.len() - 1 {
            masks.push(mask);
            mask = ndarray::Array::zeros((256, 60));
            tokens.push(token);
            token = ndarray::Array::zeros((256, 60));
        }
    }
    let encode = Instant::now();

    let file = File::create("src/rust_output.csv").unwrap();
    let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
    writer.serialize(&["output_0", "output_1"]).unwrap();

    for _ in 0..masks.len() {
        let clone = session.clone();
        let mut clone = clone.lock().unwrap();
        let result: Vec<tensor::OrtOwnedTensor<f32, ndarray::Dim<ndarray::IxDynImpl>>> =
            clone.run(vec![tokens.pop().unwrap(), masks.pop().unwrap()])?;
        result.iter().for_each(|array| {
            for row in array.outer_iter() {
                let row = row.as_slice().unwrap();
                writer.serialize(row).unwrap();
            }
        });
    }

    let write_onnx = Instant::now();
    println!("Setup: {}ms", (setup - start).as_millis());

    println!("Read: {}ms", (read - setup).as_millis());

    println!("Encode: {}ms", (encode - read).as_millis());
    println!("Write Onnx: {}s", (write_onnx - encode).as_secs());
    //   println!(
    //       "outputs: {:#?}",
    //       outputs
    //           .pop()
    //           .unwrap()
    //           .map_axis(ndarray::Axis(1), |x| x[0] > x[1])
    //           .map(|x| match x {
    //               True => "Open",
    //               False => "Not Open",
    //           })
    //   );
    //   println!("outputs: {:#?}\n", &outputs);
    // find and display the max value with its index
    Ok(())
}
