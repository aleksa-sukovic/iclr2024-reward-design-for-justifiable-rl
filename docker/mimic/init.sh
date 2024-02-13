#!/bin/bash

cd /tmp/mimic-code
make mimic-gz datadir=/var/data/mimic DBNAME=mimic DBUSER=postgres DBPASS=password DBHOST=localhost # ensure parameters match with those provided in docker-compose.yml
