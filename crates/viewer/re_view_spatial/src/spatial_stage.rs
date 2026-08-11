//! A single, in-memory Spatial 3D stage for embedders.
//!
//! This deliberately does not construct the Viewer app. It owns only the Rerun
//! state required to ingest component data and run one `SpatialView3D`: the
//! recording store, one ephemeral blueprint, the view query, and camera/picking
//! state. Hosts own their window, input session, surrounding UI, persistence,
//! and product commands.

use std::sync::Arc;

use ahash::HashMap;
use re_chunk::{Chunk, LatestAtQuery};
use re_entity_db::EntityDb;
use re_log_channel::LogReceiverSet;
use re_log_types::{ApplicationId, StoreId, StoreInfo, StoreKind, StoreSource};
use re_viewer_context::{
    AppCaches, AppContext, AppOptions, ApplicationSelectionState, CommandReceiver, CommandSender,
    ComponentUiRegistry, DragAndDropManager, FallbackProviderRegistry, FocusTarget, ItemCollection,
    MissingChunkReporter, Route, StoreHub, ViewClass as _, ViewClassRegistry, ViewId, ViewStates,
    ViewerContext, command_channel,
};
use re_viewport::execute_systems_for_view;
use re_viewport_blueprint::ViewBlueprint;

/// Rerun's Spatial 3D runtime without the Viewer application's chrome or lifecycle.
pub struct SpatialStage {
    app_options: AppOptions,
    recording_store_id: StoreId,
    store_hub: StoreHub,
    view_class_registry: ViewClassRegistry,
    app_caches: AppCaches,
    selection_state: ApplicationSelectionState,
    focused_item: Option<FocusTarget>,
    time_ctrl: re_viewer_context::TimeControl,
    blueprint_time_ctrl: re_viewer_context::TimeControl,
    view_states: ViewStates,
    blueprint_query: LatestAtQuery,
    component_ui_registry: ComponentUiRegistry,
    component_fallback_registry: FallbackProviderRegistry,
    reflection: re_types_core::reflection::Reflection,
    connection_registry: re_redap_client::ConnectionRegistryHandle,
    command_sender: CommandSender,
    command_receiver: CommandReceiver,
    view: ViewBlueprint,
    query_results: HashMap<ViewId, re_viewer_context::DataQueryResult>,
}

impl SpatialStage {
    /// Create an isolated in-memory Spatial 3D stage for one host application.
    ///
    /// The stage never loads, saves, or mutates a user-facing Rerun blueprint.
    pub fn new(application_id: ApplicationId) -> anyhow::Result<Self> {
        let app_options = AppOptions::default();
        let reflection = re_sdk_types::reflection::generate_reflection()?;
        let mut component_fallback_registry =
            re_component_fallbacks::create_component_fallback_registry();
        let mut view_class_registry = ViewClassRegistry::default();
        view_class_registry.add_class::<crate::SpatialView3D>(
            &reflection,
            &app_options,
            &mut component_fallback_registry,
        )?;

        let recording_store_id = StoreId::random(StoreKind::Recording, application_id.clone());
        let mut recording = EntityDb::new(recording_store_id.clone());
        recording.set_store_info(re_log_types::SetStoreInfo {
            row_id: *re_chunk::RowId::new(),
            info: StoreInfo::new(
                recording_store_id.clone(),
                StoreSource::Other("embedded-spatial-stage".to_owned()),
            ),
        });

        let blueprint_store_id = StoreId::random(StoreKind::Blueprint, application_id);
        let mut store_hub = StoreHub::new(Default::default(), &|_| {});
        store_hub.insert_entity_db(recording);
        store_hub.insert_entity_db(EntityDb::new(blueprint_store_id.clone()));
        store_hub.set_cloned_blueprint_active_for_app(&blueprint_store_id)?;

        let (command_sender, command_receiver) = command_channel();
        let component_ui_registry = re_component_ui::create_component_ui_registry();

        Ok(Self {
            app_options,
            recording_store_id,
            store_hub,
            view_class_registry,
            app_caches: Default::default(),
            selection_state: Default::default(),
            focused_item: None,
            time_ctrl: Default::default(),
            blueprint_time_ctrl: Default::default(),
            view_states: Default::default(),
            blueprint_query: LatestAtQuery::latest(re_viewer_context::blueprint_timeline()),
            component_ui_registry,
            component_fallback_registry,
            reflection,
            connection_registry:
                re_redap_client::ConnectionRegistry::new_without_stored_credentials(),
            command_sender,
            command_receiver,
            view: ViewBlueprint::new_with_root_wildcard(
                crate::SpatialView3D::identifier(),
            ),
            query_results: Default::default(),
        })
    }

    /// The recording store that receives this stage's Rerun component input.
    pub fn recording_store_id(&self) -> &StoreId {
        &self.recording_store_id
    }

    /// Add a translated Rerun chunk to the stage's in-memory recording.
    pub fn ingest_chunk(&mut self, chunk: Arc<Chunk>) -> anyhow::Result<()> {
        self.store_hub.add_chunk(&self.recording_store_id, &chunk)?;
        Ok(())
    }

    /// Run exactly one Spatial 3D view inside the host-provided egui region.
    ///
    /// The caller supplies the Rerun render context attached to its own device,
    /// queue, surface, and input loop. No Viewer `App`, panels, navigation,
    /// connection UI, notifications, or on-disk blueprint persistence are run.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        render_ctx: &mut re_renderer::RenderContext,
    ) -> anyhow::Result<()> {
        self.store_hub
            .begin_frame_caches(Some(&self.recording_store_id));
        self.app_caches.begin_frame();
        self.store_hub
            .entity_db_and_cache(&self.recording_store_id, &self.view_class_registry);
        render_ctx.begin_frame();

        let route = Route::LocalRecording {
            recording_id: self.recording_store_id.clone(),
        };
        let (storage_context, store_context) = self.store_hub.read_context(&route, &self.time_ctrl);
        let store_context = store_context
            .ok_or_else(|| anyhow::anyhow!("spatial stage recording is unavailable"))?;

        let visualizable_entities_per_visualizer = store_context
            .caches
            .visualizable_entities_for_visualizer_systems();
        let indicated_entities_per_visualizer =
            store_context.caches.indicated_entities_per_visualizer();
        let class = self
            .view_class_registry
            .get_class_or_log_error(self.view.class_identifier());
        let view_state =
            self.view_states
                .get_mut_or_create(&self.recording_store_id, self.view.id, class);
        let query_range = self.view.query_range(
            store_context.blueprint,
            &self.blueprint_query,
            self.time_ctrl.timeline(),
            &self.view_class_registry,
            view_state,
        );
        self.query_results.clear();
        self.query_results.insert(
            self.view.id,
            self.view.contents.build_data_result_tree(
                &store_context,
                self.time_ctrl.timeline(),
                &self.view_class_registry,
                &self.blueprint_query,
                &query_range,
                &visualizable_entities_per_visualizer,
                &indicated_entities_per_visualizer,
                &self.app_options,
            ),
        );

        let drag_and_drop_manager = DragAndDropManager::new(ItemCollection::default());
        let connected_receivers = LogReceiverSet::default();
        let egui_ctx = ui.ctx().clone();
        let ctx = ViewerContext {
            app_ctx: AppContext {
                is_test: false,
                app_options: &self.app_options,
                reflection: &self.reflection,
                egui_ctx: &egui_ctx,
                command_sender: &self.command_sender,
                render_ctx,
                connection_registry: &self.connection_registry,
                storage_context: &storage_context,
                active_store_context: Some(&store_context),
                app_caches: &self.app_caches,
                component_ui_registry: &self.component_ui_registry,
                view_class_registry: &self.view_class_registry,
                component_fallback_registry: &self.component_fallback_registry,
                route: &route,
                selection_state: &self.selection_state,
                focused_item: &self.focused_item,
                drag_and_drop_manager: &drag_and_drop_manager,
                connected_receivers: &connected_receivers,
                auth_context: None,
                login_enabled: false,
                login_signed_in_url: None,
            },
            store_context: &store_context,
            visualizable_entities_per_visualizer: &visualizable_entities_per_visualizer,
            indicated_entities_per_visualizer: &indicated_entities_per_visualizer,
            query_results: &self.query_results,
            time_ctrl: &self.time_ctrl,
            blueprint_time_ctrl: &self.blueprint_time_ctrl,
            blueprint_query: &self.blueprint_query,
        };

        self.view_states.reset_visualizer_reports();
        let context_systems = self.view_class_registry.run_once_per_frame_context_systems(
            &ctx,
            std::iter::once(self.view.class_identifier()),
        );
        let view_state =
            self.view_states
                .get_mut_or_create(&self.recording_store_id, self.view.id, class);
        let (query, system_output) =
            execute_systems_for_view(&ctx, &self.view, view_state, &context_systems);
        self.view_states.add_visualizer_reports_from_output(
            &self.recording_store_id,
            self.view.id,
            &system_output,
        );

        let missing_chunk_reporter = MissingChunkReporter::new(system_output.any_missing_chunks());
        let view_state =
            self.view_states
                .get_mut_or_create(&self.recording_store_id, self.view.id, class);
        class.ui(
            &ctx,
            &missing_chunk_reporter,
            ui,
            view_state,
            &query,
            system_output,
        )?;
        drop(context_systems);
        drop(ctx);
        render_ctx.before_submit();

        // Keep the receiver alive until the host-facing interaction translation is added.
        let _ = &self.command_receiver;
        self.selection_state
            .on_frame_start(|item| Some(item.clone()), None);
        self.focused_item = None;
        Ok(())
    }
}
