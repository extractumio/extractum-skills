# Architecture

> Last updated: __DATE__  
> Status: empty — fill in as the system takes shape.

Top-down system architecture: components, data flow, security model.

## Overview

_TBD — one-paragraph "what is this system, who uses it, how does data flow in and out."_

## Components

_TBD — one block per deployable unit. For each: language, runtime, role, upstream/downstream, where it runs, how it scales._

## Data Flow

_TBD — describe ingress, the happy-path traversal across components, and the persistence boundaries. A small ASCII diagram beats prose._

## Storage & State

_TBD — what is the source of truth for each kind of data? What is cache vs. authoritative? Backups, retention, recovery._

## Security Model

_TBD — authentication, authorization, secrets management, network boundaries, blast radius if a component is compromised._

## External Dependencies

_TBD — third-party services, APIs, hardware. SLAs and what breaks when they go down._

## Non-Goals

_TBD — what this system explicitly does **not** try to do. Saves arguments later._
