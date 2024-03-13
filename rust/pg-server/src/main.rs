/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

mod error;
mod session;
mod utility;

use crate::error::Result;
use crate::session::GqeSessionService;
use clap::Parser;
use clap_verbosity_flag::Verbosity;
use log::info;
use pgwire::api::auth::noop::NoopStartupHandler;
use pgwire::api::query::PlaceholderExtendedQueryHandler;
use pgwire::api::{MakeHandler, StatelessMakeHandler};
use pgwire::tokio::process_socket;
use sqlparser::dialect::GenericDialect;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use tokio::net::TcpListener;

static TIMER_PRECISION: usize = 3;
static SQL_DIALECT: GenericDialect = GenericDialect {};

type DefaultLogLevel = clap_verbosity_flag::InfoLevel;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct CmdArgs {
    /// IP address to listen on
    #[arg(short = 's', long, default_value_t = IpAddr::from([127, 0, 0, 1]))]
    listen_address: IpAddr,

    /// Port to listen on
    #[arg(short = 'p', long, default_value_t = 5432)]
    listen_port: u16,

    #[command(flatten)]
    verbose: Verbosity<DefaultLogLevel>,
}

fn init_logger(verbosity: &Verbosity<DefaultLogLevel>) {
    use simplelog::{ColorChoice, CombinedLogger, Config, TermLogger, TerminalMode};

    let filter = if let Some(level) = verbosity.log_level() {
        level.to_level_filter()
    } else {
        log::LevelFilter::Off
    };

    CombinedLogger::init(vec![TermLogger::new(
        filter,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )])
    .expect("Failed to set logger.");
}

#[tokio::main]
pub async fn main() {
    let cmd_args = CmdArgs::parse();

    init_logger(&cmd_args.verbose);

    let processor = Arc::new(StatelessMakeHandler::new(
        Arc::new(GqeSessionService::new()),
    ));

    // FIXME: We have not implemented extended query in this server, use placeholder instead
    let placeholder = Arc::new(StatelessMakeHandler::new(Arc::new(
        PlaceholderExtendedQueryHandler,
    )));
    let authenticator = Arc::new(StatelessMakeHandler::new(Arc::new(NoopStartupHandler)));

    let listen_address = SocketAddr::new(cmd_args.listen_address, cmd_args.listen_port);
    let listener = TcpListener::bind(listen_address).await.unwrap();

    info!("Listening to {}", listen_address.to_string());

    loop {
        let incoming_socket = listener.accept().await.unwrap();
        let authenticator_ref = authenticator.make();
        let processor_ref = processor.make();
        let placeholder_ref = placeholder.make();
        tokio::spawn(async move {
            process_socket(
                incoming_socket.0,
                None,
                authenticator_ref,
                processor_ref,
                placeholder_ref,
            )
            .await
        });
    }
}
