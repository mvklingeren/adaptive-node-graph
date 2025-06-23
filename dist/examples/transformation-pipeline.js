import { Graph, AdaptiveNode, createProcessor } from '../core';
// Validation node
const validator = createProcessor((user) => {
    if (!user.email.includes('@')) {
        throw new Error('Invalid email');
    }
    if (user.age < 0 || user.age > 150) {
        throw new Error('Invalid age');
    }
    return user;
}, 'validator');
// Enrichment node
const enricher = createProcessor((user) => ({
    ...user,
    category: user.age < 18 ? 'minor' : user.age < 65 ? 'adult' : 'senior'
}), 'enricher');
// Privacy filter
const privacyFilter = new AdaptiveNode((data) => data)
    .register(Object, (obj) => {
    const filtered = { ...obj };
    if ('email' in filtered) {
        filtered.email = filtered.email.replace(/(.{2}).*(@.*)/, '$1***$2');
    }
    return filtered;
})
    .setLabel('privacy-filter');
// Build pipeline
const graph = new Graph();
graph.addNode(validator);
graph.addNode(enricher);
graph.addNode(privacyFilter);
graph.connect(validator, enricher);
graph.connect(enricher, privacyFilter);
// Process user data
const userData = {
    id: 1,
    name: 'John Doe',
    email: 'john.doe@example.com',
    age: 25
};
const result = await graph.execute(userData, validator.id);
console.log(result);
// { id: 1, name: 'John Doe', email: 'jo***@example.com', age: 25, category: 'adult' }
